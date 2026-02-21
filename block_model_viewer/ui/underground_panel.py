"""
Underground Mining Panel - Stope Optimization & Scheduling UI

Provides graphical interface for:
- Stope optimization (maximum closure algorithm)
- Production scheduling (MILP)
- Ground control analysis
- Equipment scheduling
- Ventilation design
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QMessageBox,
    QFileDialog, QCheckBox, QSplitter, QProgressBar
)
from PyQt6.QtWidgets import QWidget
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

# Safety switch: disable DataBridge publishing if stability issues occur.
# Set to True to re-enable publishing schedule/stopes to other panels (e.g., ESG Dashboard).
ENABLE_DATABRIDGE_PUBLISH = True  # Re-enabled after stabilization; guarded publishing logic remains


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


# UGWorker removed - logic moved to controller


class UndergroundPanel(BaseAnalysisPanel):
    """Main panel for underground mining operations."""
    # PanelManager metadata
    PANEL_ID = "UndergroundPanel"
    PANEL_NAME = "Underground Panel"
    PANEL_CATEGORY = PanelCategory.PLANNING
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "underground"
    
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent=parent, panel_id="underground")
        self.blocks_df: Optional[pd.DataFrame] = None
        self.stopes: List = []
        self.schedule: List = []
        self.current_operation: Optional[str] = None  # 'optimize_stopes', 'schedule_production', etc.
        # Keep a reference to the main window (for visualization access)
        self._main_window_ref = main_window
        
        # Subscribe to block model updates from DataRegistry
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.blockModelGenerated.connect(self._on_block_model_from_registry)
                self.registry.blockModelLoaded.connect(self._on_block_model_from_registry)
                self.registry.blockModelClassified.connect(self._on_block_model_from_registry)
                
                # Prefer classified block model when available.
                existing_block_model = self.registry.get_classified_block_model()
                if not _has_model_data(existing_block_model):
                    existing_block_model = self.registry.get_block_model()
                if _has_model_data(existing_block_model):
                    self._on_block_model_from_registry(existing_block_model)
                    logger.info("Underground panel: Loaded existing block model from DataRegistry")
        except Exception as e:
            logger.warning(f"Underground panel: Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        # setup_ui() is already called by BaseAnalysisPanel._setup_base_ui() - don't call it again!
        


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
        """Initialize the user interface."""
        layout = self.main_layout
        
        # Title
        title = QLabel("Underground Mining - Stope Optimization & Scheduling")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Stope Optimization
        tabs.addTab(self._create_stope_tab(), "Stope Optimization")
        
        # Tab 2: Production Scheduling
        tabs.addTab(self._create_schedule_tab(), "Production Scheduling")
        
        # Tab 3: Ground Control
        tabs.addTab(self._create_ground_control_tab(), "Ground Control")
        
        # Tab 4: Equipment & Ventilation
        tabs.addTab(self._create_equipment_tab(), "Equipment & Ventilation")
        
        layout.addWidget(tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.status_label)
        
    def _create_stope_tab(self) -> QWidget:
        """Create stope optimization tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Data input group
        data_group = QGroupBox("Block Model Data")
        data_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_load_blocks = QPushButton("Load Block Model CSV")
        self.btn_load_blocks.clicked.connect(self._load_block_model)
        btn_layout.addWidget(self.btn_load_blocks)
        
        self.lbl_blocks_status = QLabel("No data loaded")
        btn_layout.addWidget(self.lbl_blocks_status)
        btn_layout.addStretch()
        
        data_layout.addLayout(btn_layout)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Parameters group
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QVBoxLayout()
        
        # Create parameter inputs in a grid-like layout
        params_form = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        col1.addLayout(self._create_param_row("Min NSR ($/t):", 'min_nsr', 0.0, -1000.0, 10000.0))
        col1.addLayout(self._create_param_row("Stope Width (m):", 'stope_width', 15.0, 5.0, 50.0))
        col1.addLayout(self._create_param_row("Stope Height (m):", 'stope_height', 30.0, 10.0, 100.0))
        params_form.addLayout(col1)
        
        # Column 2
        col2 = QVBoxLayout()
        col2.addLayout(self._create_param_row("Stope Length (m):", 'stope_length', 30.0, 10.0, 100.0))
        col2.addLayout(self._create_param_row("Dilution Skin (m):", 'dilution_skin', 0.5, 0.0, 5.0))
        col2.addLayout(self._create_param_row("Metal Price ($/oz):", 'metal_price', 60.0, 0.0, 5000.0))
        params_form.addLayout(col2)
        
        # Column 3
        col3 = QVBoxLayout()
        col3.addLayout(self._create_param_row("Recovery (%):", 'recovery', 85.0, 0.0, 100.0))
        col3.addLayout(self._create_param_row("Processing Cost ($/t):", 'processing_cost', 25.0, 0.0, 1000.0))
        
        self.chk_dilution = QCheckBox("Calculate Dilution")
        self.chk_dilution.setChecked(True)
        col3.addWidget(self.chk_dilution)
        
        params_form.addLayout(col3)
        params_layout.addLayout(params_form)
        
        # Run button
        self.btn_optimize = QPushButton("Run Stope Optimization")
        self.btn_optimize.clicked.connect(lambda: self._start_ug_analysis('optimize_stopes'))
        self.btn_optimize.setEnabled(False)
        params_layout.addWidget(self.btn_optimize)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results group
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout()
        
        # Summary
        self.lbl_stope_summary = QLabel("No results yet")
        results_layout.addWidget(self.lbl_stope_summary)
        
        # Results table
        self.table_stopes = QTableWidget()
        self.table_stopes.setColumnCount(8)
        self.table_stopes.setHorizontalHeaderLabels([
            "Stope ID", "Tonnes", "Grade", "NSR ($/t)", 
            "Diluted Tonnes", "Diluted Grade", "Value ($)", "Blocks"
        ])
        results_layout.addWidget(self.table_stopes)
        
        # Export and visualize buttons
        btn_export_layout = QHBoxLayout()
        self.btn_export_stopes = QPushButton("Export to CSV")
        self.btn_export_stopes.clicked.connect(self._export_stopes)
        self.btn_export_stopes.setEnabled(False)
        btn_export_layout.addWidget(self.btn_export_stopes)
        
        self.btn_visualize_stopes = QPushButton("Visualize 3D")
        self.btn_visualize_stopes.clicked.connect(self._visualize_stopes_3d)
        self.btn_visualize_stopes.setEnabled(False)
        self.btn_visualize_stopes.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_export_layout.addWidget(self.btn_visualize_stopes)
        
        btn_export_layout.addStretch()
        results_layout.addLayout(btn_export_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_schedule_tab(self) -> QWidget:
        """Create production scheduling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        info = QLabel("Production scheduling requires optimized stopes. Run stope optimization first.")
        info.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Parameters group
        params_group = QGroupBox("Scheduling Parameters")
        params_layout = QVBoxLayout()
        
        params_form = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        col1.addLayout(self._create_param_row("Number of Periods:", 'n_periods', 24, 1, 100, int_type=True))
        col1.addLayout(self._create_param_row("Mine Capacity (t/period):", 'mine_capacity', 10000.0, 0.0, 1000000.0))
        col1.addLayout(self._create_param_row("Mill Capacity (t/period):", 'mill_capacity', 10000.0, 0.0, 1000000.0))
        params_form.addLayout(col1)
        
        # Column 2
        col2 = QVBoxLayout()
        col2.addLayout(self._create_param_row("Fill Capacity (t/period):", 'fill_capacity', 8000.0, 0.0, 1000000.0))
        col2.addLayout(self._create_param_row("Discount Rate (%):", 'discount_rate', 10.0, 0.0, 50.0))
        col2.addLayout(self._create_param_row("Curing Lag (periods):", 'curing_lag', 2, 0, 10, int_type=True))
        params_form.addLayout(col2)
        
        # Column 3
        col3 = QVBoxLayout()
        col3.addLayout(self._create_param_row("Stockpile Capacity (t):", 'stockpile_capacity', 50000.0, 0.0, 1000000.0))
        
        # Sequence mode
        seq_layout = QHBoxLayout()
        seq_layout.addWidget(QLabel("Sequence Mode:"))
        self.combo_sequence = QComboBox()
        self.combo_sequence.addItems(['top_down', 'bottom_up', 'flexible'])
        seq_layout.addWidget(self.combo_sequence)
        col3.addLayout(seq_layout)
        
        # Solver
        solver_layout = QHBoxLayout()
        solver_layout.addWidget(QLabel("Solver:"))
        self.combo_solver = QComboBox()
        self.combo_solver.addItems(['glpk', 'cbc', 'cplex', 'gurobi'])
        solver_layout.addWidget(self.combo_solver)
        col3.addLayout(solver_layout)

        # Discount mode
        discount_mode_layout = QHBoxLayout()
        discount_mode_layout.addWidget(QLabel("Discount Mode:"))
        self.combo_discount_mode = QComboBox()
        self.combo_discount_mode.addItems(['annual_simple', 'monthly_compounded'])
        self.combo_discount_mode.setToolTip(
            "annual_simple: original per-period annual discounting\n"
            "monthly_compounded: converts annual rate to a per-period rate using Days/Period"
        )
        discount_mode_layout.addWidget(self.combo_discount_mode)
        col3.addLayout(discount_mode_layout)

        # Days per period (for discounting conversion)
        period_days_layout = QHBoxLayout()
        period_days_layout.addWidget(QLabel("Days / Period:"))
        self.spin_period_days = QDoubleSpinBox()
        self.spin_period_days.setRange(1.0, 365.0)
        self.spin_period_days.setDecimals(0)
        self.spin_period_days.setValue(30.0)
        self.spin_period_days.setToolTip("Used to convert annual discount rate to per-period when using monthly_compounded mode")
        period_days_layout.addWidget(self.spin_period_days)
        col3.addLayout(period_days_layout)
        
        params_form.addLayout(col3)
        params_layout.addLayout(params_form)
        
        # Run button
        self.btn_schedule = QPushButton("Run MILP Scheduling")
        self.btn_schedule.clicked.connect(lambda: self._start_ug_analysis('schedule_production'))
        self.btn_schedule.setEnabled(False)
        params_layout.addWidget(self.btn_schedule)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results group
        results_group = QGroupBox("Schedule Results")
        results_layout = QVBoxLayout()
        
        # Summary
        self.lbl_schedule_summary = QLabel("No results yet")
        results_layout.addWidget(self.lbl_schedule_summary)
        
        # Results table
        self.table_schedule = QTableWidget()
        self.table_schedule.setColumnCount(7)
        self.table_schedule.setHorizontalHeaderLabels([
            "Period", "Ore Mined (t)", "Ore Processed (t)", "Fill Placed (t)",
            "Stockpile (t)", "Cashflow ($)", "DCF ($)"
        ])
        results_layout.addWidget(self.table_schedule)
        
        # Export and visualization buttons
        btn_export_layout = QHBoxLayout()
        self.btn_export_schedule = QPushButton("Export to CSV")
        self.btn_export_schedule.clicked.connect(self._export_schedule)
        self.btn_export_schedule.setEnabled(False)
        btn_export_layout.addWidget(self.btn_export_schedule)
        
        self.btn_view_gantt = QPushButton("View Gantt Chart")
        self.btn_view_gantt.clicked.connect(self._view_gantt_chart)
        self.btn_view_gantt.setEnabled(False)
        self.btn_view_gantt.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        btn_export_layout.addWidget(self.btn_view_gantt)
        
        btn_export_layout.addStretch()
        results_layout.addLayout(btn_export_layout)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_ground_control_tab(self) -> QWidget:
        """Create ground control analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Parameters group
        params_group = QGroupBox("Rock Mass Properties")
        params_layout = QVBoxLayout()
        
        params_form = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        col1.addLayout(self._create_param_row("UCS (MPa):", 'ucs', 80.0, 0.1, 500.0))
        col1.addLayout(self._create_param_row("RQD (%):", 'rqd', 75.0, 0.0, 100.0))
        col1.addLayout(self._create_param_row("Joint Spacing (m):", 'spacing', 0.3, 0.01, 10.0))
        params_form.addLayout(col1)
        
        # Column 2
        col2 = QVBoxLayout()
        col2.addLayout(self._create_param_row("Joint Condition (0-6):", 'condition', 3, 0, 6, int_type=True))
        col2.addLayout(self._create_param_row("Groundwater (0-15):", 'groundwater', 10, 0, 15, int_type=True))
        col2.addLayout(self._create_param_row("Orientation (-12 to 0):", 'orientation', -5, -12, 0, int_type=True))
        params_form.addLayout(col2)
        
        # Column 3 (Q-system parameters)
        col3 = QVBoxLayout()
        col3.addLayout(self._create_param_row("Jn (joint sets):", 'jn', 9, 0.5, 20, int_type=True))
        col3.addLayout(self._create_param_row("Jr (roughness):", 'jr', 3, 0.5, 4, int_type=True))
        col3.addLayout(self._create_param_row("Ja (alteration):", 'ja', 2, 0.75, 20, int_type=True))
        params_form.addLayout(col3)
        
        params_layout.addLayout(params_form)
        
        # Pillar analysis
        pillar_layout = QHBoxLayout()
        self.chk_pillar = QCheckBox("Include Pillar Analysis")
        pillar_layout.addWidget(self.chk_pillar)
        
        pillar_layout.addWidget(QLabel("Pillar Width (m):"))
        self.spin_pillar_width = QDoubleSpinBox()
        self.spin_pillar_width.setRange(1.0, 50.0)
        self.spin_pillar_width.setValue(6.0)
        pillar_layout.addWidget(self.spin_pillar_width)
        
        pillar_layout.addWidget(QLabel("Pillar Height (m):"))
        self.spin_pillar_height = QDoubleSpinBox()
        self.spin_pillar_height.setRange(1.0, 20.0)
        self.spin_pillar_height.setValue(3.0)
        pillar_layout.addWidget(self.spin_pillar_height)
        
        pillar_layout.addStretch()
        params_layout.addLayout(pillar_layout)
        
        # Run button
        self.btn_ground_control = QPushButton("Run Ground Control Analysis")
        self.btn_ground_control.clicked.connect(lambda: self._start_ug_analysis('analyze_ground_control'))
        params_layout.addWidget(self.btn_ground_control)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Results group
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        self.text_ground_results = QTextEdit()
        self.text_ground_results.setReadOnly(True)
        self.text_ground_results.setPlaceholderText("Results will appear here...")
        results_layout.addWidget(self.text_ground_results)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_equipment_tab(self) -> QWidget:
        """Create equipment and ventilation tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        info = QLabel("Equipment and ventilation calculations require a production schedule.")
        info.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Equipment section
        equip_group = QGroupBox("Equipment Requirements")
        equip_layout = QVBoxLayout()
        
        equip_params = QHBoxLayout()
        equip_params.addLayout(self._create_param_row("Haul Distance (m):", 'haul_distance', 500.0, 0.0, 5000.0))
        equip_params.addLayout(self._create_param_row("Hours/Day:", 'hours_per_day', 16.0, 1.0, 24.0))
        equip_params.addLayout(self._create_param_row("Period Days:", 'equip_period_days', 30, 1, 365, int_type=True))  # FIXED: was 'period_days' (duplicate)
        equip_layout.addLayout(equip_params)
        
        self.btn_equipment = QPushButton("Calculate Equipment")
        self.btn_equipment.clicked.connect(lambda: self._start_ug_analysis('schedule_equipment'))
        self.btn_equipment.setEnabled(False)
        equip_layout.addWidget(self.btn_equipment)
        
        self.text_equipment = QTextEdit()
        self.text_equipment.setReadOnly(True)
        self.text_equipment.setMaximumHeight(150)
        equip_layout.addWidget(self.text_equipment)
        
        equip_group.setLayout(equip_layout)
        layout.addWidget(equip_group)
        
        # Ventilation section
        vent_group = QGroupBox("Ventilation Design")
        vent_layout = QVBoxLayout()
        
        vent_params = QHBoxLayout()
        vent_params.addLayout(self._create_param_row("Required Airflow (m³/s):", 'required_airflow', 250.0, 1.0, 5000.0))
        vent_params.addLayout(self._create_param_row("Total Resistance (Ns²/m⁸):", 'total_resistance', 0.05, 0.0, 10.0))
        vent_layout.addLayout(vent_params)
        
        vent_params2 = QHBoxLayout()
        vent_params2.addLayout(self._create_param_row("Fan Efficiency (%):", 'fan_efficiency', 75.0, 1.0, 100.0))
        vent_params2.addLayout(self._create_param_row("Pressure Margin (Pa):", 'pressure_margin', 500.0, 0.0, 5000.0))  # Added missing spinner
        vent_layout.addLayout(vent_params2)
        
        self.btn_ventilation = QPushButton("Design Ventilation System")
        self.btn_ventilation.clicked.connect(lambda: self._start_ug_analysis('design_ventilation'))
        vent_layout.addWidget(self.btn_ventilation)
        
        self.text_ventilation = QTextEdit()
        self.text_ventilation.setReadOnly(True)
        self.text_ventilation.setMaximumHeight(150)
        vent_layout.addWidget(self.text_ventilation)
        
        vent_group.setLayout(vent_layout)
        layout.addWidget(vent_group)
        
        layout.addStretch()
        
        return widget
        
    def _create_param_row(self, label: str, attr_name: str, default: float, 
                         min_val: float, max_val: float, int_type: bool = False) -> QHBoxLayout:
        """Create a parameter input row."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        
        if int_type:
            spin = QSpinBox()
            spin.setRange(int(min_val), int(max_val))
            spin.setValue(int(default))
        else:
            spin = QDoubleSpinBox()
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            spin.setDecimals(2)
        
        setattr(self, f'spin_{attr_name}', spin)
        layout.addWidget(spin)
        
        return layout
        
    def _load_block_model(self):
        """Load block model from CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Block Model CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)

                # Try to normalise using BlockModel if available
                try:
                    from ..models.block_model import BlockModel
                    bm = BlockModel()
                    bm.update_from_dataframe(df)
                    if bm.block_count == 0:
                        raise ValueError("No valid block geometry found in file")
                    # Convert canonical dataframe
                    self.blocks_df = bm.to_dataframe()
                except Exception:
                    # Fallback: attempt to accept several common coordinate names
                    lowered = [c.lower() for c in df.columns]
                    if all(name in lowered for name in ['x', 'y', 'z']):
                        # Keep ALL columns; just ensure x,y,z exist (lowercase for internal consistency)
                        # Identify the original casing of coordinate columns
                        coord_map = {}
                        for c in df.columns:
                            lc = c.lower()
                            if lc in ('x','y','z') and lc not in coord_map:
                                coord_map[lc] = c
                        self.blocks_df = df.copy()
                        # Create lowercase duplicates if necessary for x,y,z expected downstream
                        for lc, orig in coord_map.items():
                            if orig != lc:
                                self.blocks_df[lc] = self.blocks_df[orig]
                    else:
                        QMessageBox.warning(
                            self, "Invalid Data",
                            "Block model must contain coordinate columns. Supported sets: XC/YC/ZC, XMORIG/YMORIG/ZMORIG, x/y/z, or grid XINC/YINC/ZINC."
                        )
                        return

                self.lbl_blocks_status.setText(
                    f"Loaded: {len(self.blocks_df)} blocks, {len(self.blocks_df.columns)} attributes"
                )
                self.btn_optimize.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading File", str(e))
                
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def _start_ug_analysis(self, operation: str) -> None:
        """Start underground mining analysis with specified operation."""
        self.current_operation = operation
        self.run_analysis()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI based on current operation."""
        if self.blocks_df is None or self.blocks_df.empty:
            raise ValueError("Please load a block model first.")
        
        if self.current_operation == "optimize_stopes":
            return {
                "operation": "optimize_stopes",
                "blocks_df": self.blocks_df.copy(),
                "min_nsr": self.spin_min_nsr.value(),
                "stope_width": self.spin_stope_width.value(),
                "stope_height": self.spin_stope_height.value(),
                "stope_length": self.spin_stope_length.value(),
                "dilution_skin": self.spin_dilution_skin.value(),
                "metal_price": self.spin_metal_price.value(),
                "recovery": self.spin_recovery.value() / 100.0,
                "processing_cost": self.spin_processing_cost.value(),
                "calculate_dilution": self.chk_dilution.isChecked(),
            }
        elif self.current_operation == "schedule_production":
            if not self.stopes:
                raise ValueError("Please optimize stopes first.")
            return {
                "operation": "schedule_production",
                "stopes": self.stopes,
                "mine_capacity": self.spin_mine_capacity.value(),
                "mill_capacity": self.spin_mill_capacity.value(),
                "fill_capacity": self.spin_fill_capacity.value(),
                "n_periods": self.spin_n_periods.value(),
                "discount_rate": self.spin_discount_rate.value() / 100.0,
                "curing_lag": self.spin_curing_lag.value(),
                "stockpile_capacity": self.spin_stockpile_capacity.value(),
                "sequence_mode": self.combo_sequence.currentText(),  # FIXED: was combo_sequence_mode
                "solver": self.combo_solver.currentText(),
                "discount_mode": self.combo_discount_mode.currentText(),
                "period_days": self.spin_period_days.value(),
            }
        elif self.current_operation == "analyze_ground_control":
            return {
                "operation": "analyze_ground_control",
                "ucs": self.spin_ucs.value(),
                "spacing": self.spin_spacing.value(),
                "groundwater": self.spin_groundwater.value(),
                "rqd": self.spin_rqd.value(),
                "jn": self.spin_jn.value(),
                "jr": self.spin_jr.value(),  # FIXED: was jw (doesn't exist)
                "ja": self.spin_ja.value(),  # ADDED: missing Q-system param
                "condition": self.spin_condition.value(),  # ADDED: needed for RMR
                "orientation": self.spin_orientation.value(),  # ADDED: needed for RMR
                "jw": 1.0,  # Default water factor (not in UI)
                "calculate_pillar": self.chk_pillar.isChecked(),  # FIXED: was chk_calculate_pillar
                "pillar_width": self.spin_pillar_width.value(),
                "pillar_height": self.spin_pillar_height.value(),
            }
        elif self.current_operation == "schedule_equipment":
            if not self.schedule:
                raise ValueError("Please run production scheduling first.")
            return {
                "operation": "schedule_equipment",
                "schedule": self.schedule,
                "haul_distance": self.spin_haul_distance.value(),
                "hours_per_day": self.spin_hours_per_day.value(),
                "period_days": self.spin_equip_period_days.value(),  # FIXED: was spin_period_days (duplicate)
            }
        elif self.current_operation == "design_ventilation":
            return {
                "operation": "design_ventilation",
                "required_airflow": self.spin_required_airflow.value(),
                "total_resistance": self.spin_total_resistance.value(),
                "fan_efficiency": self.spin_fan_efficiency.value() / 100.0,
                "pressure_margin": self.spin_pressure_margin.value(),
            }
        else:
            raise ValueError(f"Unknown underground operation: {self.current_operation}")
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.blocks_df is None or self.blocks_df.empty:
            self.show_error("No Block Model", "Please load a block model first.")
            return False
        
        if self.current_operation == "schedule_production" and not self.stopes:
            self.show_error("No Stopes", "Please optimize stopes first.")
            return False
        
        if self.current_operation == "schedule_equipment" and not self.schedule:
            self.show_error("No Schedule", "Please run production scheduling first.")
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display underground mining analysis results."""
        operation = payload.get("operation")
        results = payload.get("results", {})
        
        if operation == "optimize_stopes":
            self._handle_stope_results(results)
        elif operation == "schedule_production":
            self._handle_schedule_results(results)
        elif operation == "analyze_ground_control":
            self._handle_ground_control_results(results)
        elif operation == "schedule_equipment":
            self._handle_equipment_results(results)
        elif operation == "design_ventilation":
            self._handle_ventilation_results(results)
    
    def _run_stope_optimization(self):
        """Run stope optimization (legacy method - now routed through BaseAnalysisPanel)."""
        # This method is kept for backward compatibility but is no longer used
        # Button connections now call _start_ug_analysis('optimize_stopes')
        pass
        
    def _handle_stope_results(self, results: dict):
        """Handle stope optimization results."""
        self.stopes = results['stopes']
        self.blocks_df = results['blocks_df']
        diagnostics = results.get('diagnostics', {})
        if diagnostics:
            try:
                logger.info(f"UG stope optimization diagnostics: {diagnostics}")
            except Exception:
                pass
        
        # Update summary
        if self.stopes:
            total_tonnes = sum(s.diluted_tonnes if hasattr(s, 'diluted_tonnes') else s.tonnes for s in self.stopes)
            avg_grade = sum(
                (s.diluted_grade if hasattr(s, 'diluted_grade') else s.grade) * 
                (s.diluted_tonnes if hasattr(s, 'diluted_tonnes') else s.tonnes) 
                for s in self.stopes
            ) / total_tonnes if total_tonnes > 0 else 0
            total_value = sum(s.nsr * (s.diluted_tonnes if hasattr(s, 'diluted_tonnes') else s.tonnes) for s in self.stopes)
            
            self.lbl_stope_summary.setText(
                f"Optimized {len(self.stopes)} stopes | "
                f"Total: {total_tonnes:,.0f} t | "
                f"Avg Grade: {avg_grade:.2f} | "
                f"Value: ${total_value:,.0f}"
            )
        else:
            reason = ""
            if diagnostics:
                blocks_above = diagnostics.get('blocks_above_threshold')
                if blocks_above is not None:
                    reason += f" Blocks >= min NSR: {blocks_above}."
                grade_source = diagnostics.get('grade_source')
                if grade_source:
                    reason += f" Grade source: {grade_source}."
                nsr_stats = diagnostics.get('nsr_stats_pre_filter')
                if nsr_stats:
                    reason += f" NSR(min/mean/max): {nsr_stats.get('min'):.2f}/{nsr_stats.get('mean'):.2f}/{nsr_stats.get('max'):.2f}."
                thresh = diagnostics.get('min_nsr_threshold')
                if thresh is not None:
                    reason += f" Threshold: {thresh:.2f}."
            self.lbl_stope_summary.setText("No viable stopes found." + reason)
            self.status_label.setText("No stopes above threshold. Adjust min NSR or verify grade column.")
        
        # Populate table
        self.table_stopes.setRowCount(len(self.stopes))
        for i, stope in enumerate(self.stopes):
            # Basic columns
            self.table_stopes.setItem(i, 0, QTableWidgetItem(stope.id))
            self.table_stopes.setItem(i, 1, QTableWidgetItem(f"{stope.tonnes:,.0f}"))
            self.table_stopes.setItem(i, 2, QTableWidgetItem(f"{stope.grade:.3f}"))
            self.table_stopes.setItem(i, 3, QTableWidgetItem(f"{stope.nsr:.2f}"))

            # Dilution columns (if present)
            if hasattr(stope, 'diluted_tonnes') and stope.diluted_tonnes is not None:
                self.table_stopes.setItem(i, 4, QTableWidgetItem(f"{stope.diluted_tonnes:,.0f}"))
                self.table_stopes.setItem(i, 5, QTableWidgetItem(f"{stope.diluted_grade:.3f}"))
                value = stope.nsr * stope.diluted_tonnes
            else:
                self.table_stopes.setItem(i, 4, QTableWidgetItem("-"))
                self.table_stopes.setItem(i, 5, QTableWidgetItem("-"))
                value = stope.nsr * stope.tonnes

            # Value and block count
            self.table_stopes.setItem(i, 6, QTableWidgetItem(f"${value:,.0f}"))
            # Use block_indices from Stope dataclass
            block_count = len(getattr(stope, 'block_indices', getattr(stope, 'block_ids', [])))
            self.table_stopes.setItem(i, 7, QTableWidgetItem(str(block_count)))

        self.table_stopes.resizeColumnsToContents()

        # Enable export, visualize, and scheduling
        self.btn_export_stopes.setEnabled(True)
        self.btn_visualize_stopes.setEnabled(True)
        if len(self.stopes) > 0:
            self.btn_schedule.setEnabled(True)

        self.btn_optimize.setEnabled(True)
        self.status_label.setText("Optimization complete")
    
    def _on_block_model_from_registry(self, block_model):
        """
        Handle block model received from DataRegistry.
        
        This method is called when a block model is loaded or generated elsewhere
        in the application and makes it available for underground mining analysis.
        """
        logger.info(f"Underground panel: Received block model from DataRegistry: {type(block_model)}")
        
        success = self.set_block_model(block_model)
        
        if success:
            logger.info(f"Underground panel: Successfully set block model with {len(self.blocks_df)} blocks")
        else:
            logger.warning("Underground panel: Failed to set block model from DataRegistry")
    
    def set_block_model(self, block_model_or_df):
        """Accept a BlockModel instance or a pandas DataFrame and set internal blocks_df.

        This makes the panel robust to block models produced elsewhere in the app
        (BlockModel objects from the renderer / builder) as well as raw DataFrames
        using different coordinate column naming conventions.
        """
        try:
            # If it's a BlockModel, convert to DataFrame canonical form
            from ..models.block_model import BlockModel
            if isinstance(block_model_or_df, BlockModel):
                df = block_model_or_df.to_dataframe()
            elif isinstance(block_model_or_df, (pd.DataFrame,)):
                # Try to normalise via BlockModel.update_from_dataframe
                bm = BlockModel()
                bm.update_from_dataframe(block_model_or_df)
                if bm.block_count == 0:
                    raise ValueError("Provided DataFrame does not contain valid block geometry")
                df = bm.to_dataframe()
            else:
                # Try to coerce list/dict to DataFrame
                df = pd.DataFrame(block_model_or_df)
                bm = BlockModel()
                bm.update_from_dataframe(df)
                df = bm.to_dataframe()

            # Final validation
            if df.empty or not all(col in df.columns for col in ['x', 'y', 'z']):
                raise ValueError("Normalized block model does not contain x,y,z columns")

            self.blocks_df = df
            self.lbl_blocks_status.setText(f"Loaded: {len(self.blocks_df)} blocks (from app)")
            self.btn_optimize.setEnabled(True)
            logger.info("UndergroundPanel: block model set via set_block_model()")
            return True
        except Exception as e:
            logger.warning(f"UndergroundPanel: Failed to set block model: {e}")
            return False
        
    def _run_scheduling(self):
        """Run production scheduling."""
        logger.info("=== Main: _run_scheduling called ===")
        
        if not self.stopes:
            logger.warning("Main: No stopes available")
            QMessageBox.warning(self, "No Stopes", "Please run stope optimization first.")
            return
        
        logger.info(f"Main: Starting scheduling with {len(self.stopes)} stopes")
        self.btn_schedule.setEnabled(False)
        self.status_label.setText("Scheduling production...")
        
        kwargs = {
            'stopes': self.stopes,
            'n_periods': self.spin_n_periods.value(),
            'mine_capacity': self.spin_mine_capacity.value(),
            'mill_capacity': self.spin_mill_capacity.value(),
            'fill_capacity': self.spin_fill_capacity.value(),
            'discount_rate': self.spin_discount_rate.value() / 100.0,
            'curing_lag': self.spin_curing_lag.value(),
            'stockpile_capacity': self.spin_stockpile_capacity.value(),
            'sequence_mode': self.combo_sequence.currentText(),
            'solver': self.combo_solver.currentText(),
            'discount_mode': self.combo_discount_mode.currentText(),
            'period_days': float(self.spin_period_days.value()),
        }
        logger.info(f"Main: Config - periods={kwargs['n_periods']}, mine_cap={kwargs['mine_capacity']}")
        
        # Legacy method - now uses BaseAnalysisPanel.run_analysis() via _start_ug_analysis
        # This method is kept for backward compatibility but is no longer used
        logger.warning("_run_schedule_production is deprecated - use _start_ug_analysis('schedule_production')")
        self._start_ug_analysis('schedule_production')
        
    def _handle_schedule_results(self, results: dict):
        """Handle scheduling results."""
        logger.info("=== Main: _handle_schedule_results called ===")
        try:
            self.schedule = results['schedule']
            logger.info(f"Main: Received {len(self.schedule)} schedule periods")
            
            # Calculate summary statistics (always, even if empty)
            total_mined = 0.0
            total_processed = 0.0
            total_dcf = 0.0
            total_available = 0.0
            
            # Update summary
            if self.schedule:
                logger.info("Main: Calculating summary statistics...")
                total_mined = sum(p.ore_mined for p in self.schedule)
                total_processed = sum(p.ore_proc for p in self.schedule)
                total_dcf = sum(p.dcf for p in self.schedule)
                # Compute available tonnes from stopes for coverage metric
                try:
                    total_available = sum(
                        (getattr(s, 'diluted_tonnes') if getattr(s, 'diluted_tonnes', None) is not None else getattr(s, 'tonnes', 0.0))
                        for s in (self.stopes or [])
                    )
                except Exception:
                    total_available = 0.0
                scheduled_pct = (total_mined / total_available * 100.0) if total_available > 0 else 0.0
                logger.info(
                    f"Main: Summary - mined={total_mined:,.0f}t, processed={total_processed:,.0f}t, NPV=${total_dcf:,.0f}, scheduled={scheduled_pct:.1f}% of available"
                )
                
                self.lbl_schedule_summary.setText(
                    f"{len(self.schedule)} periods | "
                    f"Mined: {total_mined:,.0f} t | "
                    f"Processed: {total_processed:,.0f} t | "
                    f"NPV: ${total_dcf:,.0f} | "
                    f"Coverage: {scheduled_pct:.1f}%"
                )
            else:
                logger.warning("Main: No schedule generated")
                self.lbl_schedule_summary.setText("No schedule generated")
            
            # Populate table
            logger.info("Main: Populating schedule table...")
            self.table_schedule.setRowCount(len(self.schedule))
            for i, period in enumerate(self.schedule):
                self.table_schedule.setItem(i, 0, QTableWidgetItem(str(period.t)))
                self.table_schedule.setItem(i, 1, QTableWidgetItem(f"{period.ore_mined:,.0f}"))
                self.table_schedule.setItem(i, 2, QTableWidgetItem(f"{period.ore_proc:,.0f}"))
                self.table_schedule.setItem(i, 3, QTableWidgetItem(f"{period.fill_placed:,.0f}"))
                self.table_schedule.setItem(i, 4, QTableWidgetItem(f"{period.stockpile:,.0f}"))
                self.table_schedule.setItem(i, 5, QTableWidgetItem(f"${period.cashflow:,.0f}"))
                self.table_schedule.setItem(i, 6, QTableWidgetItem(f"${period.dcf:,.0f}"))
            
            self.table_schedule.resizeColumnsToContents()
            logger.info("Main: Table populated successfully")
            
            # Publish schedule to DataBridge for ESG Dashboard (guarded by safety switch)
            if ENABLE_DATABRIDGE_PUBLISH:
                logger.info("Main: Attempting to publish to DataBridge...")
                try:
                    from ..utils.data_bridge import get_data_bridge, DataType
                    bridge = get_data_bridge()
                    logger.info("Main: DataBridge imported successfully")
                    
                    # Publish schedule with metadata
                    metadata = {
                        'total_periods': len(self.schedule),
                        'total_mined': total_mined,
                        'total_processed': total_processed,
                        'npv': total_dcf,
                        'timestamp': self.schedule[0].t if self.schedule else 0
                    }
                    logger.info(f"Main: Publishing schedule with metadata: {metadata}")
                    
                    bridge.publish(
                        DataType.PRODUCTION_SCHEDULE,
                        self.schedule,
                        source="UndergroundMining",
                        metadata=metadata
                    )
                    logger.info("Main: Schedule published successfully")
                    
                    # Also publish stopes if available
                    if self.stopes:
                        logger.info(f"Main: Publishing {len(self.stopes)} stopes...")
                        bridge.publish(
                            DataType.STOPE_LIST,
                            self.stopes,
                            source="UndergroundMining",
                            metadata={'total_stopes': len(self.stopes)}
                        )
                        logger.info("Main: Stopes published successfully")
                    
                    logger.info("Main: All data published to DataBridge")
                except Exception as e:
                    logger.error(f"Main: Could not publish to DataBridge: {e}", exc_info=True)
            else:
                logger.info("Main: Skipping DataBridge publishing (ENABLE_DATABRIDGE_PUBLISH=False)")
            
            # Publish schedule to DataRegistry
            try:
                if hasattr(self, 'registry') and self.registry:
                    schedule_data = {
                        'schedule': self.schedule,
                        'total_mined': total_mined,
                        'total_processed': total_processed,
                        'npv': total_dcf
                    }
                    self.registry.register_schedule(schedule_data, source_panel="UndergroundPanel")
                    logger.info("Published underground schedule to DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to publish underground schedule to DataRegistry: {e}")
            
            # Enable export, Gantt, and equipment buttons
            logger.info("Main: Re-enabling buttons...")
            self.btn_export_schedule.setEnabled(True)
            self.btn_view_gantt.setEnabled(True)
            self.btn_equipment.setEnabled(True)
            
            self.btn_schedule.setEnabled(True)
            self.status_label.setText("Scheduling complete")
            logger.info("=== Main: _handle_schedule_results complete ===")
            
        except Exception as e:
            logger.error(f"Main: EXCEPTION in _handle_schedule_results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to process results:\n{e}")
            self.btn_schedule.setEnabled(True)
        
    def _run_ground_control(self):
        """Run ground control analysis."""
        self.btn_ground_control.setEnabled(False)
        self.status_label.setText("Analyzing ground control...")
        
        kwargs = {
            'ucs': self.spin_ucs.value(),
            'rqd': self.spin_rqd.value(),
            'spacing': self.spin_spacing.value(),
            'condition': self.spin_condition.value(),
            'groundwater': self.spin_groundwater.value(),
            'orientation': self.spin_orientation.value(),
            'jn': self.spin_jn.value(),
            'jr': self.spin_jr.value(),
            'ja': self.spin_ja.value(),
            'calculate_pillar': self.chk_pillar.isChecked(),
            'pillar_width': self.spin_pillar_width.value(),
            'pillar_height': self.spin_pillar_height.value()
        }
        
        # Worker removed - logic moved to controller
        # This code is now handled by BaseAnalysisPanel.run_analysis()
        pass
        
    def _handle_ground_control_results(self, results: dict):
        """Handle ground control results."""
        rmr = results.get('rmr')
        q = results.get('q_value')
        pillar_fos = results.get('pillar_fos')
        support = results.get('support')  # may be a string (placeholder API) or a dict (future implementation)
        notes = results.get('properties').notes if results.get('properties') is not None else ""
        
        # Format results
        output = "=" * 60 + "\n"
        output += "GROUND CONTROL ANALYSIS RESULTS\n"
        output += "=" * 60 + "\n\n"
        
        output += f"Rock Mass Rating (RMR): {rmr}\n"
        
        # RMR classification
        if rmr >= 81:
            rmr_class = "VERY GOOD"
        elif rmr >= 61:
            rmr_class = "GOOD"
        elif rmr >= 41:
            rmr_class = "FAIR"
        elif rmr >= 21:
            rmr_class = "POOR"
        else:
            rmr_class = "VERY POOR"
        output += f"RMR Classification: {rmr_class}\n\n"
        
        if q is not None:
            output += f"Q-System Value: {q:.2f}\n"
        
        # Q classification
        if q is not None:
            if q > 40:
                q_class = "Exceptionally Good"
            elif q > 10:
                q_class = "Very Good"
            elif q > 4:
                q_class = "Good"
            elif q > 1:
                q_class = "Fair"
            elif q > 0.1:
                q_class = "Poor"
            elif q > 0.01:
                q_class = "Very Poor"
            else:
                q_class = "Extremely Poor"
            output += f"Q Classification: {q_class}\n\n"

        if pillar_fos is not None:
            output += f"Pillar Factor of Safety: {pillar_fos:.2f}\n"
            if pillar_fos >= 1.5:
                output += "Pillar Status: SAFE\n\n"
            elif pillar_fos >= 1.0:
                output += "Pillar Status: MARGINAL - Monitor closely\n\n"
            else:
                output += "Pillar Status: UNSAFE - Redesign required\n\n"
        
        output += "Recommended Ground Support:\n"
        output += "-" * 60 + "\n"
        if isinstance(support, dict):
            for key, value in support.items():
                output += f"{key}: {value}\n"
        elif isinstance(support, (list, tuple)):
            for item in support:
                output += f"- {item}\n"
        elif isinstance(support, str):
            output += f"{support}\n"
        else:
            output += "(No support recommendations available)\n"

        if notes:
            output += "\nNotes: " + notes + "\n"
        
        self.text_ground_results.setPlainText(output)
        
        self.btn_ground_control.setEnabled(True)
        self.status_label.setText("Ground control analysis complete")
        
    def _run_equipment(self):
        """Calculate equipment requirements."""
        if not self.schedule:
            QMessageBox.warning(self, "No Schedule", "Please run production scheduling first.")
            return
        
        self.btn_equipment.setEnabled(False)
        self.status_label.setText("Calculating equipment...")
        
        kwargs = {
            'schedule': self.schedule,
            'haul_distance': self.spin_haul_distance.value(),
            'hours_per_day': self.spin_hours_per_day.value(),
            'period_days': self.spin_period_days.value()
        }
        
        # Worker removed - logic moved to controller
        # This code is now handled by BaseAnalysisPanel.run_analysis()
        pass
        
    def _handle_equipment_results(self, results: dict):
        """Handle equipment results."""
        equipment = results.get('equipment')

        output = "EQUIPMENT REQUIREMENTS\n"
        output += "=" * 60 + "\n\n"

        if isinstance(equipment, dict):
            for eq_type, count in equipment.items():
                output += f"{eq_type}: {count} units\n"
        elif isinstance(equipment, (list, tuple)):
            for item in equipment:
                output += f"- {item}\n"
        elif isinstance(equipment, str):
            output += equipment + "\n"
        else:
            output += "(No equipment requirements available)\n"

        self.text_equipment.setPlainText(output)
        self.btn_equipment.setEnabled(True)
        self.status_label.setText("Equipment calculation complete")
        
    def _run_ventilation(self):
        """Design ventilation system."""
        self.btn_ventilation.setEnabled(False)
        self.status_label.setText("Designing ventilation...")
        
        kwargs = {
            'required_airflow': self.spin_required_airflow.value(),
            'total_resistance': self.spin_total_resistance.value(),
            'fan_efficiency': self.spin_fan_efficiency.value() / 100.0
        }
        
        # Worker removed - logic moved to controller
        # This code is now handled by BaseAnalysisPanel.run_analysis()
        pass
        
    def _handle_ventilation_results(self, results: dict):
        """Handle ventilation results."""
        vent = results.get('ventilation')

        output = "VENTILATION SYSTEM DESIGN\n"
        output += "=" * 60 + "\n\n"

        if isinstance(vent, dict):
            airflow = vent.get('airflow')
            pressure = vent.get('pressure')
            power = vent.get('power')
            operating_cost = vent.get('operating_cost')
            if airflow is not None:
                output += f"Required Airflow: {airflow:.1f} m³/s\n"
            if pressure is not None:
                output += f"Design Pressure: {pressure:.0f} Pa\n"
            if power is not None:
                output += f"Fan Power: {power:.0f} kW\n"
            if operating_cost is not None:
                output += f"Operating Cost: ${operating_cost:.0f}/year\n"
        elif isinstance(vent, (list, tuple)):
            for item in vent:
                output += f"- {item}\n"
        elif isinstance(vent, str):
            output += vent + "\n"
        else:
            output += "(No ventilation results available)\n"

        self.text_ventilation.setPlainText(output)
        self.btn_ventilation.setEnabled(True)
        self.status_label.setText("Ventilation design complete")
        
    def _export_stopes(self):
        """Export stopes to CSV."""
        if not self.stopes:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Stopes CSV", "stopes_optimized.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            # Create DataFrame
            data = []
            for stope in self.stopes:
                row = {
                    'stope_id': stope.id,
                    'tonnes': stope.tonnes,
                    'grade': stope.grade,
                    'nsr': stope.nsr,
                    'num_blocks': len(stope.block_ids)
                }
                if hasattr(stope, 'diluted_tonnes'):
                    row['diluted_tonnes'] = stope.diluted_tonnes
                    row['diluted_grade'] = stope.diluted_grade
                data.append(row)
            
            df = pd.DataFrame(data)
            # Step 10: Use ExportHelpers
            from ..utils.export_helpers import export_dataframe_to_csv
            export_dataframe_to_csv(df, file_path)
            
            QMessageBox.information(self, "Export Complete", f"Stopes exported to {file_path}")
            
    def _export_schedule(self):
        """Export schedule to CSV."""
        if not self.schedule:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Schedule CSV", "production_schedule.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            data = []
            for period in self.schedule:
                data.append({
                    'period': period.t,
                    'ore_mined': period.ore_mined,
                    'ore_processed': period.ore_proc,
                    'fill_placed': period.fill_placed,
                    'stockpile': period.stockpile,
                    'cashflow': period.cashflow,
                    'dcf': period.dcf
                })
            
            df = pd.DataFrame(data)
            # Step 10: Use ExportHelpers
            from ..utils.export_helpers import export_dataframe_to_csv
            export_dataframe_to_csv(df, file_path)
            
            QMessageBox.information(self, "Export Complete", f"Schedule exported to {file_path}")
    
    def _visualize_stopes_3d(self):
        """Send stopes to main viewer for 3D visualization."""
        if not self.stopes or self.blocks_df is None:
            QMessageBox.warning(self, "No Data", "No stopes available to visualize.")
            return
        
        # Show progress for visualization
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Preparing 3D visualization...")
        
        try:
            from ..visualization.stope_visualizer import StopeVisualizer
            
            self.progress_bar.setValue(10)
            
            # Locate renderer via controller (PyVista isolation)
            renderer = None
            main_window = None
            if self.controller and hasattr(self.controller, 'r'):
                renderer = self.controller.r
            elif self._main_window_ref is not None and hasattr(self._main_window_ref, 'viewer_widget'):
                main_window = self._main_window_ref
                renderer = getattr(self._main_window_ref.viewer_widget, 'renderer', None)
            # Walk parent chain if needed
            if renderer is None:
                p = self.parent()
                while p is not None and renderer is None:
                    if hasattr(p, 'viewer_widget'):
                        main_window = p
                        renderer = getattr(p.viewer_widget, 'renderer', None)
                        if renderer is not None:
                            break
                    p = p.parent() if hasattr(p, 'parent') else None
            # Fallback to top-level widgets
            if renderer is None:
                from PyQt6.QtWidgets import QApplication
                for tl in QApplication.topLevelWidgets():
                    if hasattr(tl, 'viewer_widget'):
                        renderer = getattr(tl.viewer_widget, 'renderer', None)
                        if renderer is not None:
                            main_window = tl
                            self._main_window_ref = tl  # cache for next time
                            break
            if renderer is None or renderer.plotter is None:
                QMessageBox.warning(
                    self,
                    "No Viewer",
                    "Cannot access main viewer widget.\n\n"
                    "Please open the Underground panel from the main window, or ensure the viewer is initialized."
                )
                return
            
            plotter = renderer.plotter  # Get plotter from renderer for StopeVisualizer
            
            # Create visualizer
            viz = StopeVisualizer()
            self.progress_bar.setValue(25)
            
            # Determine color-by option
            color_options = ['NSR', 'Grade', 'Period (if scheduled)']
            from PyQt6.QtWidgets import QInputDialog
            
            color_by, ok = QInputDialog.getItem(
                self,
                "Stope Visualization",
                "Color stopes by:",
                color_options,
                0,
                False
            )
            
            if not ok:
                self.progress_bar.setVisible(False)
                return
            
            self.progress_bar.setValue(35)
            
            # Map selection to attribute
            if 'NSR' in color_by:
                attr = 'nsr'
                cmap = 'RdYlGn'
            elif 'Grade' in color_by:
                attr = 'grade'
                cmap = 'plasma'
            else:  # Period
                attr = 'period'
                cmap = 'tab20'
                if not self.schedule:
                    QMessageBox.warning(
                        self,
                        "No Schedule",
                        "Cannot color by period without a production schedule.\n\n"
                        "Please run scheduling first or choose another color option."
                    )
                    return
            
            # Add stopes to plotter
            self.status_label.setText("Creating 3D visualization...")
            self.progress_bar.setValue(50)
            
            actor_name = viz.add_stopes_to_plotter(
                plotter=plotter,
                stopes=self.stopes,
                blocks_df=self.blocks_df,
                color_by=attr,
                schedule=self.schedule if self.schedule else None,
                show_edges=True,
                opacity=0.8,
                cmap=cmap
            )
            
            self.progress_bar.setValue(90)
            
            if actor_name:
                # VIOLATION FIX: Update legend via LegendManager after adding stopes
                if hasattr(renderer, 'legend_manager') and renderer.legend_manager is not None:
                    try:
                        import numpy as np
                        # Get scalar values from stopes for legend range
                        if attr == 'nsr':
                            values = np.array([s.nsr for s in self.stopes])
                            finite_values = values[np.isfinite(values)]
                            if len(finite_values) > 0:
                                renderer.legend_manager.update_continuous(
                                    property_name=attr.upper(),
                                    data=values,
                                    cmap_name=cmap,
                                    vmin=float(np.nanmin(finite_values)),
                                    vmax=float(np.nanmax(finite_values))
                                )
                        elif attr == 'grade':
                            values = np.array([s.grade for s in self.stopes])
                            finite_values = values[np.isfinite(values)]
                            if len(finite_values) > 0:
                                renderer.legend_manager.update_continuous(
                                    property_name=attr.upper(),
                                    data=values,
                                    cmap_name=cmap,
                                    vmin=float(np.nanmin(finite_values)),
                                    vmax=float(np.nanmax(finite_values))
                                )
                        elif attr == 'period':
                            # For period, use categorical legend
                            if self.schedule:
                                period_map = {}
                                for period in self.schedule:
                                    if hasattr(period, 'stopes_mined'):
                                        for stope_id in period.stopes_mined:
                                            period_map[stope_id] = period.t
                                period_values = [period_map.get(s.id, 0) for s in self.stopes]
                                unique_periods = sorted(set(period_values))
                                renderer.legend_manager.update_discrete(
                                    property_name=attr.upper(),
                                    categories=unique_periods,
                                    cmap_name=cmap
                                )
                    except Exception as e:
                        logger.debug(f"Could not update legend for stopes: {e}")
                
                # Reset camera to fit all
                plotter.reset_camera()
                plotter.render()
                
                self.progress_bar.setValue(100)
                self.status_label.setText(f"Stopes visualized: {len(self.stopes)} stopes")
                QMessageBox.information(
                    self,
                    "Visualization Complete",
                    f"Successfully added {len(self.stopes)} stopes to the 3D viewer.\n\n"
                    f"Colored by: {color_by}\n"
                    f"Actor name: {actor_name}\n\n"
                    f"You can now interact with the stopes in the main viewer."
                )
            else:
                self.status_label.setText("Visualization failed")
                QMessageBox.warning(self, "Visualization Failed", "Could not create stope visualization.")
            
            self.progress_bar.setVisible(False)
                
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"Stope visualization requires PyVista:\n\n{str(e)}\n\n"
                f"Install with: pip install pyvista"
            )
        except Exception as e:
            logger.error(f"Error visualizing stopes: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Error creating 3D visualization:\n\n{str(e)}"
            )
            self.status_label.setText("Visualization error")
    
    def _view_gantt_chart(self):
        """Display interactive Gantt chart for production schedule."""
        if not self.schedule or not self.stopes:
            QMessageBox.warning(self, "No Data", "No schedule available. Please run scheduling first.")
            return
        
        try:
            from ..visualization.gantt_chart import GanttChart
            from PyQt6.QtWidgets import QInputDialog
            import matplotlib.pyplot as plt
            
            # Ask user which chart type to display
            chart_types = ['Detailed Schedule (all stopes)', 'Period Summary (aggregate)']
            chart_type, ok = QInputDialog.getItem(
                self,
                "Gantt Chart Type",
                "Select chart type:",
                chart_types,
                0,
                False
            )
            
            if not ok:
                return
            
            # Convert schedule to dict format (period -> stopes)
            schedule_dict = {}
            for period in self.schedule:
                if hasattr(period, 'stopes_mined') and period.stopes_mined:
                    schedule_dict[period.t] = period.stopes_mined
            
            if not schedule_dict:
                QMessageBox.warning(
                    self,
                    "No Stopes",
                    "Schedule does not contain stope assignments.\n\n"
                    "This may occur if scheduling failed or no stopes were scheduled."
                )
                return
            
            # Build stope attributes dict
            stope_attrs = {}
            for stope in self.stopes:
                stope_id = stope.stope_id if hasattr(stope, 'stope_id') else id(stope)
                stope_attrs[stope_id] = {
                    'nsr': getattr(stope, 'nsr', 0),
                    'grade': getattr(stope, 'grade', 0),
                    'tonnes': getattr(stope, 'tonnes', 0)
                }
            
            # Create Gantt chart
            gantt = GanttChart(figsize=(16, 10))
            
            self.status_label.setText("Generating Gantt chart...")
            
            if 'Detailed' in chart_type:
                # Ask for coloring option
                color_options = ['Period', 'NSR', 'Grade', 'Tonnes']
                color_by, ok = QInputDialog.getItem(
                    self,
                    "Color Scheme",
                    "Color bars by:",
                    color_options,
                    0,
                    False
                )
                
                if not ok:
                    color_by = 'Period'
                
                color_attr = color_by.lower()
                
                fig = gantt.create_schedule_gantt(
                    schedule=schedule_dict,
                    stope_attributes=stope_attrs,
                    color_by=color_attr,
                    title='Underground Mining Production Schedule - Detailed View',
                    period_duration_days=90,
                    show_labels=True,
                    colormap='tab20' if color_attr == 'period' else 'plasma'
                )
            else:
                # Period summary
                fig = gantt.create_period_summary_gantt(
                    schedule=schedule_dict,
                    stope_attributes=stope_attrs,
                    title='Underground Mining Production Schedule - Period Summary',
                    period_duration_days=90,
                    show_stats=True
                )
            
            if fig:
                plt.show()
                self.status_label.setText("Gantt chart displayed")
                
                # Ask if user wants to save
                from PyQt6.QtWidgets import QFileDialog
                save_option = QMessageBox.question(
                    self,
                    "Save Chart",
                    "Would you like to save this Gantt chart?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if save_option == QMessageBox.StandardButton.Yes:
                    file_path, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save Gantt Chart",
                        "",
                        "PNG Images (*.png);;PDF Files (*.pdf);;All Files (*.*)"
                    )
                    
                    if file_path:
                        gantt.save_chart(file_path)
                        QMessageBox.information(
                            self,
                            "Chart Saved",
                            f"Gantt chart saved to:\n{file_path}"
                        )
            else:
                self.status_label.setText("Failed to create Gantt chart")
                QMessageBox.warning(self, "Chart Error", "Could not generate Gantt chart.")
                
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"Gantt chart requires matplotlib:\n\n{str(e)}\n\n"
                f"Install with: pip install matplotlib"
            )
        except Exception as e:
            logger.error(f"Error creating Gantt chart: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Chart Error",
                f"Error creating Gantt chart:\n\n{str(e)}"
            )
            self.status_label.setText("Gantt chart error")
            
    def _update_status(self, message: str):
        """Update status label."""
        self.status_label.setText(message)
    
    def _update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        
    def _handle_error(self, error_msg: str):
        """Handle worker errors."""
        QMessageBox.critical(self, "Error", f"An error occurred:\n\n{error_msg}")
        self.status_label.setText("Error occurred")
        
        # Re-enable buttons
        self.btn_optimize.setEnabled(True)
        self.btn_schedule.setEnabled(True)
        self.btn_ground_control.setEnabled(True)
        self.btn_equipment.setEnabled(True)
        self.btn_ventilation.setEnabled(True)
    
    def closeEvent(self, event):
        """Clean up worker thread on close."""
        try:
            # Cancel any running tasks via controller
            if self.controller and hasattr(self.controller, 'cancel_task'):
                self.controller.cancel_task(self.task_name)
        except Exception as e:
            logger.warning(f"Error stopping worker thread: {e}")
        
        super().closeEvent(event)
