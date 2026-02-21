"""
ESG Dashboard Panel - Environmental, Social, and Governance Metrics

Provides comprehensive ESG tracking and reporting:
- Carbon footprint tracking (Scope 1, 2, 3)
- Water balance monitoring
- Waste and land management
- Multi-framework compliance (GRI, ICMM, TCFD, SASB)
"""

from __future__ import annotations

# Matplotlib backend is set in main.py before any imports

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import io

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QMessageBox, QFileDialog, QProgressBar,
    QComboBox, QDoubleSpinBox, QSpinBox, QScrollArea, QDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPixmap

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)

try:
    # Backend already set to Agg above

    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception as exc:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - charts will be disabled: %s", exc)


# ESGWorker removed - logic moved to controller


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding charts."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        if not MATPLOTLIB_AVAILABLE:
            super().__init__(Figure(figsize=(width, height), dpi=dpi))
            return
            
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()


class ESGDashboardPanel(BaseAnalysisPanel):
    """Main ESG dashboard panel."""
    
    task_name = "esg"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="esg")
        try:
            logger.info("Initializing ESGDashboardPanel...")
            self.schedule = None
            self.carbon_data = None
            self.water_data = None
            self.waste_data = None
            self.reports = None
            self.current_operation: Optional[str] = None  # 'carbon_footprint', 'water_balance', 'waste_tracking', 'generate_reports'
            
            # Subscribe to production data from DataRegistry
            try:
                self.registry = self.get_registry()
                # Note: ESG data signals can be added as needed
                # For now, connect to schedule for production data
                self.registry.scheduleGenerated.connect(self._on_schedule_generated)
                self.registry.reconciliationResultsLoaded.connect(self._on_reconciliation_loaded)
                
                # Load existing data if available
                existing_schedule = self.registry.get_schedule()
                if existing_schedule:
                    self._on_schedule_generated(existing_schedule)
                
                existing_reconciliation = self.registry.get_reconciliation_results()
                if existing_reconciliation:
                    self._on_reconciliation_loaded(existing_reconciliation)
            except Exception as e:
                logger.warning(f"Failed to connect to DataRegistry: {e}")
                self.registry = None
            self.data_bridge = None
            
            logger.info("Starting UI initialization...")
            # Initialize UI FIRST before setting up data bridge
            self.setup_ui()
            logger.info("UI initialization complete")
            
            logger.info("Scheduling DataBridge setup (deferred)...")
            # Defer DataBridge setup to next event loop cycle to avoid potential init race/Qt issues
            try:
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, self._setup_data_bridge)
            except Exception as defer_err:
                logger.error(f"Failed to defer DataBridge setup: {defer_err}")
                # Fallback to immediate attempt
                self._setup_data_bridge()
            logger.info("DataBridge setup request queued")
            
            logger.info("ESGDashboardPanel initialization complete")
        except Exception as e:
            logger.error(f"ESGDashboardPanel initialization FAILED: {e}", exc_info=True)
            raise
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("ESG Dashboard received schedule from DataRegistry")
        self.schedule = schedule
        # Update ESG metrics based on production schedule
        # Implementation depends on schedule format
    
    def _on_reconciliation_loaded(self, reconciliation_results):
        """
        Automatically receive reconciliation results when they're loaded.
        
        Args:
            reconciliation_results: Reconciliation results from DataRegistry
        """
        logger.info("ESG Dashboard received reconciliation results from DataRegistry")
        # Update ESG metrics based on actual production data
        # Implementation depends on reconciliation format
    
    def _setup_data_bridge(self):
        """Setup DataBridge connection for inter-panel communication."""
        try:
            logger.info("Importing DataBridge...")
            from ..utils.data_bridge import get_data_bridge, DataType
            logger.info("Getting DataBridge instance...")
            bridge = get_data_bridge()
            if bridge is None:
                raise RuntimeError("get_data_bridge returned None")
            self.data_bridge = bridge
            logger.info("DataBridge instance obtained")

            # Connect to schedule updates signal
            logger.info("Connecting to schedule_updated signal...")
            try:
                # Use Qt queued connection implicitly (cross-thread safe)
                self.data_bridge.schedule_updated.connect(self._on_schedule_received)
                logger.info("DataBridge schedule_updated signal connected")
            except Exception as conn_err:
                logger.error("Failed to connect schedule_updated: %s", conn_err, exc_info=True)

            # Check if schedule already available
            logger.info("Checking for existing schedule in DataBridge...")
            try:
                if self.data_bridge.has(DataType.PRODUCTION_SCHEDULE):
                    schedule = self.data_bridge.get(DataType.PRODUCTION_SCHEDULE)
                    logger.info(
                        "Found existing schedule in DataBridge (len=%s)",
                        len(schedule) if hasattr(schedule, '__len__') else 'N/A'
                    )
                    if schedule:
                        self._on_schedule_received(schedule)
                        logger.info("Loaded existing schedule from DataBridge")
                else:
                    logger.info("No schedule currently in DataBridge")
            except Exception as check_err:
                logger.error("Error checking existing schedule: %s", check_err, exc_info=True)

        except Exception as e:
            logger.error(f"Could not setup DataBridge: {e}", exc_info=True)
            self.data_bridge = None
    
    def _on_schedule_received(self, schedule):
        """Handle schedule received from DataBridge."""
        try:
            logger.info(f"_on_schedule_received called with schedule type: {type(schedule)}")
            self.schedule = schedule
            
            # Update UI
            if self.schedule:
                schedule_len = len(self.schedule) if hasattr(self.schedule, '__len__') else 'unknown'
                logger.info(f"Updating UI with schedule length: {schedule_len}")
                
                self.lbl_schedule_status.setText(
                    f"✓ Auto-loaded: {schedule_len} periods from Underground Mining"
                )
                self.lbl_schedule_status.setStyleSheet(
                    "background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724;"
                )
                
                # Enable calculation buttons
                self.btn_calc_carbon.setEnabled(True)
                self.btn_calc_water.setEnabled(True)
                self.btn_calc_waste.setEnabled(True)
                
                self.status_label.setText("Schedule received from Underground Mining panel")
                logger.info(f"ESG Dashboard auto-loaded schedule: {schedule_len} periods")
            else:
                self.lbl_schedule_status.setText("No schedule available")
                self.lbl_schedule_status.setStyleSheet(
                    "background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24;"
                )
        except Exception as e:
            logger.error(f"Error in _on_schedule_received: {e}", exc_info=True)
        
    def setup_ui(self):
        """Initialize user interface."""
        logger.debug("Setting up ESG dashboard UI layout")
        layout = self.main_layout
        
        # Title
        title = QLabel("ESG Dashboard - Environmental, Social & Governance")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Tab 1: Carbon Footprint
        tabs.addTab(self._create_carbon_tab(), "Carbon Footprint")
        
        # Tab 2: Water Balance
        tabs.addTab(self._create_water_tab(), "Water Management")
        
        # Tab 3: Waste & Land
        tabs.addTab(self._create_waste_tab(), "Waste & Land")
        
        # Tab 4: Compliance Reports
        tabs.addTab(self._create_reports_tab(), "Compliance Reports")
        
        layout.addWidget(tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_label = QLabel("Ready - Load production schedule to begin")
        self.status_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
        layout.addWidget(self.status_label)
        logger.debug("ESG dashboard UI setup finished")
        
    def _create_carbon_tab(self) -> QWidget:
        """Create carbon footprint tab."""
        logger.debug("Creating carbon footprint tab")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Load schedule button
        btn_layout = QHBoxLayout()
        self.btn_load_schedule = QPushButton("Load Production Schedule")
        self.btn_load_schedule.clicked.connect(self._load_schedule)
        btn_layout.addWidget(self.btn_load_schedule)
        
        self.btn_refresh_bridge = QPushButton("🔄 Refresh from Underground")
        self.btn_refresh_bridge.clicked.connect(self._refresh_from_bridge)
        self.btn_refresh_bridge.setStyleSheet("background-color: #17a2b8; color: white; font-weight: bold;")
        self.btn_refresh_bridge.setToolTip("Get latest schedule from Underground Mining panel")
        btn_layout.addWidget(self.btn_refresh_bridge)
        
        self.lbl_schedule_status = QLabel("No schedule loaded")
        btn_layout.addWidget(self.lbl_schedule_status)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Calculate button
        self.btn_calc_carbon = QPushButton("Calculate Carbon Footprint")
        self.btn_calc_carbon.clicked.connect(self._calculate_carbon)
        self.btn_calc_carbon.setEnabled(False)
        layout.addWidget(self.btn_calc_carbon)
        
        # Results group
        results_group = QGroupBox("Carbon Footprint Results")
        results_layout = QVBoxLayout()
        
        # KPI cards
        kpi_layout = QHBoxLayout()
        
        self.lbl_total_co2e = self._create_kpi_card("Total Emissions", "0 t CO2e", "#ff6b6b")
        kpi_layout.addWidget(self.lbl_total_co2e)
        
        self.lbl_intensity = self._create_kpi_card("Intensity", "0 kg/t", "#4ecdc4")
        kpi_layout.addWidget(self.lbl_intensity)
        
        self.lbl_scope1 = self._create_kpi_card("Scope 1", "0 t", "#45b7d1")
        kpi_layout.addWidget(self.lbl_scope1)
        
        self.lbl_scope2 = self._create_kpi_card("Scope 2", "0 t", "#96ceb4")
        kpi_layout.addWidget(self.lbl_scope2)
        
        results_layout.addLayout(kpi_layout)
        
        # Chart
        carbon_canvas = self._create_matplotlib_canvas(width=8, height=4, name="carbon")
        if carbon_canvas is not None:
            self.carbon_canvas = carbon_canvas
            results_layout.addWidget(carbon_canvas)
        else:
            results_layout.addWidget(QLabel("Install matplotlib for charts"))
        
        # Details table
        self.table_carbon = QTableWidget()
        self.table_carbon.setColumnCount(3)
        self.table_carbon.setHorizontalHeaderLabels(["Activity", "CO2e (tonnes)", "Percentage"])
        self.table_carbon.setMaximumHeight(150)
        results_layout.addWidget(self.table_carbon)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_water_tab(self) -> QWidget:
        """Create water management tab."""
        logger.debug("Creating water management tab")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info
        info = QLabel("Water balance simulation requires production schedule.")
        info.setStyleSheet("background-color: #d1ecf1; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Parameters
        params_group = QGroupBox("Water System Parameters")
        params_layout = QVBoxLayout()
        
        # Create parameter inputs
        params_form = QHBoxLayout()
        
        col1 = QVBoxLayout()
        col1.addLayout(self._create_param("Pit Capacity (m³):", 'pit_capacity', 50000.0))
        col1.addLayout(self._create_param("Pit Inflow (m³/period):", 'pit_inflow', 100.0))
        params_form.addLayout(col1)
        
        col2 = QVBoxLayout()
        col2.addLayout(self._create_param("Tailings Capacity (m³):", 'tailings_capacity', 500000.0))
        col2.addLayout(self._create_param("Tailings Area (ha):", 'tailings_area', 50.0))
        params_form.addLayout(col2)
        
        col3 = QVBoxLayout()
        col3.addLayout(self._create_param("Period Days:", 'period_days', 30, int_type=True))
        params_form.addLayout(col3)
        
        params_layout.addLayout(params_form)
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Calculate and visualization buttons
        btn_layout = QHBoxLayout()
        
        self.btn_calc_water = QPushButton("Simulate Water Balance")
        self.btn_calc_water.clicked.connect(lambda: self._start_esg_analysis('water_balance'))
        self.btn_calc_water.setEnabled(False)
        btn_layout.addWidget(self.btn_calc_water)
        
        self.btn_view_sankey = QPushButton("View Sankey Diagram")
        self.btn_view_sankey.clicked.connect(self._view_water_sankey)
        self.btn_view_sankey.setEnabled(False)
        self.btn_view_sankey.setStyleSheet("background-color: #16a085; color: white; font-weight: bold;")
        btn_layout.addWidget(self.btn_view_sankey)
        
        layout.addLayout(btn_layout)
        
        # Results
        results_group = QGroupBox("Water Balance Results")
        results_layout = QVBoxLayout()
        
        # KPIs
        kpi_layout = QHBoxLayout()
        
        self.lbl_water_use = self._create_kpi_card("Total Use", "0 m³", "#3498db")
        kpi_layout.addWidget(self.lbl_water_use)
        
        self.lbl_recycling = self._create_kpi_card("Recycling Rate", "0%", "#2ecc71")
        kpi_layout.addWidget(self.lbl_recycling)
        
        self.lbl_compliance = self._create_kpi_card("Compliance", "0%", "#f39c12")
        kpi_layout.addWidget(self.lbl_compliance)
        
        results_layout.addLayout(kpi_layout)
        
        # Chart
        water_canvas = self._create_matplotlib_canvas(width=8, height=4, name="water")
        if water_canvas is not None:
            self.water_canvas = water_canvas
            results_layout.addWidget(water_canvas)
        else:
            results_layout.addWidget(QLabel("Install matplotlib for charts"))
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_waste_tab(self) -> QWidget:
        """Create waste and land tab."""
        logger.debug("Creating waste & land tab")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Parameters
        params_group = QGroupBox("Waste Parameters")
        params_layout = QHBoxLayout()
        
        params_layout.addLayout(self._create_param("Strip Ratio:", 'strip_ratio', 3.0))
        params_layout.addLayout(self._create_param("PAG Percentage:", 'pag_percentage', 15.0))
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Calculate button
        self.btn_calc_waste = QPushButton("Calculate Waste & Disturbance")
        self.btn_calc_waste.clicked.connect(lambda: self._start_esg_analysis('waste_tracking'))
        self.btn_calc_waste.setEnabled(False)
        layout.addWidget(self.btn_calc_waste)
        
        # Results
        results_group = QGroupBox("Waste & Land Results")
        results_layout = QVBoxLayout()
        
        # KPIs
        kpi_layout = QHBoxLayout()
        
        self.lbl_waste_total = self._create_kpi_card("Total Waste", "0 t", "#e74c3c")
        kpi_layout.addWidget(self.lbl_waste_total)
        
        self.lbl_pag_waste = self._create_kpi_card("PAG Waste", "0 t", "#c0392b")
        kpi_layout.addWidget(self.lbl_pag_waste)
        
        self.lbl_disturbed = self._create_kpi_card("Disturbed Area", "0 ha", "#e67e22")
        kpi_layout.addWidget(self.lbl_disturbed)
        
        self.lbl_rehab = self._create_kpi_card("Rehab Area", "0 ha", "#27ae60")
        kpi_layout.addWidget(self.lbl_rehab)
        
        results_layout.addLayout(kpi_layout)
        
        # Chart
        if MATPLOTLIB_AVAILABLE:
            self.waste_canvas = MplCanvas(self, width=8, height=4)
            results_layout.addWidget(self.waste_canvas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)
        
        return widget
        
    def _create_reports_tab(self) -> QWidget:
        """Create compliance reports tab."""
        logger.debug("Creating compliance reports tab")
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info
        info = QLabel("Generate ESG compliance reports after calculating carbon, water, and waste metrics.")
        info.setStyleSheet("background-color: #fff3cd; padding: 10px; border-radius: 5px;")
        layout.addWidget(info)
        
        # Generate button
        self.btn_generate_reports = QPushButton("Generate ESG Reports")
        self.btn_generate_reports.clicked.connect(lambda: self._start_esg_analysis('generate_reports'))
        self.btn_generate_reports.setEnabled(False)
        layout.addWidget(self.btn_generate_reports)
        
        # Compliance scores
        scores_group = QGroupBox("Compliance Scores")
        scores_layout = QHBoxLayout()
        
        self.lbl_gri_score = self._create_kpi_card("GRI", "0%", "#9b59b6")
        scores_layout.addWidget(self.lbl_gri_score)
        
        self.lbl_tcfd_score = self._create_kpi_card("TCFD", "0 disclosures", "#8e44ad")
        scores_layout.addWidget(self.lbl_tcfd_score)
        
        self.lbl_sasb_score = self._create_kpi_card("SASB", "0%", "#3498db")
        scores_layout.addWidget(self.lbl_sasb_score)
        
        scores_group.setLayout(scores_layout)
        layout.addWidget(scores_group)
        
        # Reports text
        reports_group = QGroupBox("Report Details")
        reports_layout = QVBoxLayout()
        
        # Framework selector
        framework_layout = QHBoxLayout()
        framework_layout.addWidget(QLabel("Framework:"))
        self.combo_framework = QComboBox()
        self.combo_framework.addItems(['GRI', 'TCFD', 'SASB'])
        self.combo_framework.currentTextChanged.connect(self._display_report)
        framework_layout.addWidget(self.combo_framework)
        framework_layout.addStretch()
        
        # Export button
        self.btn_export_report = QPushButton("Export to JSON")
        self.btn_export_report.clicked.connect(self._export_reports)
        self.btn_export_report.setEnabled(False)
        framework_layout.addWidget(self.btn_export_report)
        
        reports_layout.addLayout(framework_layout)
        
        # Report text
        self.text_report = QTextEdit()
        self.text_report.setReadOnly(True)
        self.text_report.setPlaceholderText("Report details will appear here...")
        reports_layout.addWidget(self.text_report)
        
        reports_group.setLayout(reports_layout)
        layout.addWidget(reports_group, stretch=1)
        
        return widget
        
    def _create_kpi_card(self, title: str, value: str, color: str) -> QLabel:
        """Create a KPI display card."""
        label = QLabel(f"<b>{title}</b><br><span style='font-size:18pt'>{value}</span>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"""
            background-color: {color};
            color: white;
            padding: 15px;
            border-radius: 5px;
            min-width: 120px;
        """)
        return label
        
    def _create_param(self, label: str, attr: str, default: float, 
                     int_type: bool = False) -> QHBoxLayout:
        """Create parameter input row."""
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label))
        
        if int_type:
            spin = QSpinBox()
            spin.setRange(1, 1000000)
            spin.setValue(int(default))
        else:
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000000.0)
            spin.setValue(default)
            spin.setDecimals(2)
        
        setattr(self, f'spin_{attr}', spin)
        layout.addWidget(spin)
        
        return layout

    def _create_matplotlib_canvas(self, width: int, height: int, name: str) -> Optional['MplCanvas']:
        """Safely create a Matplotlib canvas and log failures."""
        if not MATPLOTLIB_AVAILABLE:
            logger.info("Matplotlib unavailable - skipping %s canvas", name)
            return None
        try:
            canvas = MplCanvas(self, width=width, height=height)
            logger.debug("Created %s Matplotlib canvas", name)
            return canvas
        except Exception as exc:
            logger.error("Failed to create %s Matplotlib canvas: %s", name, exc, exc_info=True)
            return None
        
    def _load_schedule(self):
        """Load production schedule from file or memory."""
        # For now, show message to use Underground panel first
        QMessageBox.information(
            self,
            "Load Schedule",
            "Production schedule can be generated in the Underground Mining panel.\n\n"
            "For now, you can also load a schedule CSV file with columns: "
            "period, ore_mined, ore_proc, fill_placed"
        )
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Schedule CSV", "", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                df = pd.read_csv(file_path)
                
                # Convert to PeriodKPI objects
                from ..ug import PeriodKPI
                
                self.schedule = []
                for _, row in df.iterrows():
                    self.schedule.append(PeriodKPI(
                        t=int(row.get('period', row.get('t', 0))),
                        ore_mined=float(row.get('ore_mined', 0)),
                        ore_proc=float(row.get('ore_proc', row.get('ore_processed', 0))),
                        fill_placed=float(row.get('fill_placed', 0)),
                        stockpile=float(row.get('stockpile', 0)),
                        cashflow=float(row.get('cashflow', 0)),
                        dcf=float(row.get('dcf', 0))
                    ))
                
                self.lbl_schedule_status.setText(f"Loaded: {len(self.schedule)} periods")
                self.btn_calc_carbon.setEnabled(True)
                self.btn_calc_water.setEnabled(True)
                self.btn_calc_waste.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Schedule", str(e))
    
    def _refresh_from_bridge(self):
        """Manually refresh data from DataBridge."""
        if not hasattr(self, 'data_bridge') or self.data_bridge is None:
            QMessageBox.warning(
                self,
                "DataBridge Not Available",
                "DataBridge connection not established.\n\n"
                "Cannot automatically sync with Underground Mining panel."
            )
            return
        
        try:
            from ..utils.data_bridge import DataType
            
            # Check if schedule available
            if self.data_bridge.has(DataType.PRODUCTION_SCHEDULE):
                package = self.data_bridge.get_package(DataType.PRODUCTION_SCHEDULE)
                
                if package:
                    self.schedule = package.data
                    
                    # Update UI with metadata
                    metadata = package.metadata
                    periods = metadata.get('total_periods', len(self.schedule))
                    npv = metadata.get('npv', 0)
                    
                    self.lbl_schedule_status.setText(
                        f"✓ Synced: {periods} periods | NPV: ${npv:,.0f} | "
                        f"Source: {package.source} | v{package.version}"
                    )
                    self.lbl_schedule_status.setStyleSheet(
                        "background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724;"
                    )
                    
                    # Enable buttons
                    self.btn_calc_carbon.setEnabled(True)
                    self.btn_calc_water.setEnabled(True)
                    self.btn_calc_waste.setEnabled(True)
                    
                    self.status_label.setText(f"Refreshed from DataBridge (v{package.version})")
                    
                    QMessageBox.information(
                        self,
                        "Data Refreshed",
                        f"Successfully loaded schedule from {package.source}:\n\n"
                        f"Periods: {periods}\n"
                        f"NPV: ${npv:,.0f}\n"
                        f"Version: {package.version}\n"
                        f"Updated: {package.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:
                    QMessageBox.warning(self, "No Data", "No schedule data available in DataBridge.")
            else:
                QMessageBox.information(
                    self,
                    "No Schedule Available",
                    "No production schedule found in DataBridge.\n\n"
                    "Please run scheduling in the Underground Mining panel first."
                )
        
        except Exception as e:
            logger.error(f"Error refreshing from DataBridge: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Refresh Error",
                f"Error refreshing data from DataBridge:\n\n{str(e)}"
            )
                
    def _calculate_carbon(self):
        """Calculate carbon footprint - now uses BaseAnalysisPanel.run_analysis()."""
        # Legacy method - now routed through _start_esg_analysis('carbon_footprint')
        self._start_esg_analysis('carbon_footprint')
        
    def _handle_carbon_results(self, results: dict):
        """Handle carbon calculation results."""
        self.carbon_data = results
        
        # Update KPIs
        total_co2e = results['total_co2e']
        intensity = results['intensity']
        
        self._update_kpi_card(self.lbl_total_co2e, "Total Emissions", f"{total_co2e:,.0f} t CO2e")
        self._update_kpi_card(self.lbl_intensity, "Intensity", f"{intensity:.1f} kg/t")
        self._update_kpi_card(self.lbl_scope1, "Scope 1", f"{total_co2e * 0.3:,.0f} t")
        self._update_kpi_card(self.lbl_scope2, "Scope 2", f"{total_co2e * 0.7:,.0f} t")
        
        # Update table
        summary = results['summary']
        self.table_carbon.setRowCount(len(summary))
        for i, row in summary.iterrows():
            self.table_carbon.setItem(i, 0, QTableWidgetItem(row['activity']))
            self.table_carbon.setItem(i, 1, QTableWidgetItem(f"{row['co2e_tonnes']:,.0f}"))
            pct = row['co2e_tonnes'] / total_co2e * 100
            self.table_carbon.setItem(i, 2, QTableWidgetItem(f"{pct:.1f}%"))
        
        # Update chart
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'carbon_canvas'):
            self.carbon_canvas.axes.clear()
            
            # Pie chart
            self.carbon_canvas.axes.pie(
                summary['co2e_tonnes'],
                labels=summary['activity'],
                autopct='%1.1f%%',
                startangle=90
            )
            self.carbon_canvas.axes.set_title("Carbon Footprint by Activity")
            self.carbon_canvas.draw()
        
        self.btn_calc_carbon.setEnabled(True)
        self.btn_generate_reports.setEnabled(True)
        self.status_label.setText("Carbon footprint calculated")
        
    def _simulate_water(self):
        """Simulate water balance (legacy method - now routed through BaseAnalysisPanel)."""
        # This method is kept for backward compatibility but is no longer used
        # Button connections now call _start_esg_analysis('water_balance')
        pass
        
    def _handle_water_results(self, results: dict):
        """Handle water balance results."""
        self.water_data = results
        
        # Update KPIs
        total_use = results['total_use']
        recycling_rate = results['recycling_rate']
        compliance_rate = results['compliance_rate']
        
        self._update_kpi_card(self.lbl_water_use, "Total Use", f"{total_use:,.0f} m³")
        self._update_kpi_card(self.lbl_recycling, "Recycling Rate", f"{recycling_rate*100:.1f}%")
        self._update_kpi_card(self.lbl_compliance, "Compliance", f"{compliance_rate*100:.0f}%")
        
        # Update chart
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'water_canvas'):
            self.water_canvas.axes.clear()
            
            # Time series of water use and recycling
            periods = [r.period for r in results['results']]
            water_use = [r.water_use for r in results['results']]
            recycled = [r.recycled for r in results['results']]
            
            self.water_canvas.axes.plot(periods, water_use, label='Total Use', marker='o')
            self.water_canvas.axes.plot(periods, recycled, label='Recycled', marker='s')
            self.water_canvas.axes.set_xlabel('Period')
            self.water_canvas.axes.set_ylabel('Water (m³)')
            self.water_canvas.axes.set_title('Water Use Over Time')
            self.water_canvas.axes.legend()
            self.water_canvas.axes.grid(True, alpha=0.3)
            self.water_canvas.draw()
        
        self.btn_calc_water.setEnabled(True)
        self.btn_view_sankey.setEnabled(True)
        self.btn_generate_reports.setEnabled(True)
        self.status_label.setText("Water balance simulated")
        
    def _track_waste(self):
        """Track waste production (legacy method - now routed through BaseAnalysisPanel)."""
        # This method is kept for backward compatibility but is no longer used
        # Button connections now call _start_esg_analysis('waste_tracking')
        pass
        
    def _handle_waste_results(self, results: dict):
        """Handle waste tracking results."""
        self.waste_data = results
        
        waste_report = results['waste_report']
        disturbance = results['disturbance']
        rehab_area = results['rehab_area']
        
        # Calculate PAG waste
        pag_waste = sum(
            waste_report.waste_by_type.get('PAG', 0)
            for _ in [waste_report]
        )
        
        # Update KPIs
        self._update_kpi_card(
            self.lbl_waste_total,
            "Total Waste",
            f"{waste_report.waste_generated_t:,.0f} t"
        )
        self._update_kpi_card(
            self.lbl_pag_waste,
            "PAG Waste",
            f"{pag_waste:,.0f} t"
        )
        self._update_kpi_card(
            self.lbl_disturbed,
            "Disturbed Area",
            f"{disturbance.disturbed_area_ha:.1f} ha"
        )
        self._update_kpi_card(
            self.lbl_rehab,
            "Rehab Area",
            f"{rehab_area:.1f} ha"
        )
        
        # Store total waste for reports
        if not hasattr(self, 'waste_data') or self.waste_data is None:
            self.waste_data = {}
        self.waste_data['total_waste'] = waste_report.waste_generated_t
        self.waste_data['tailings'] = 0  # Simplified
        
        self.btn_calc_waste.setEnabled(True)
        self.btn_generate_reports.setEnabled(True)
        self.status_label.setText("Waste tracking complete")
        
    def _generate_reports(self):
        """Generate ESG reports (legacy method - now routed through BaseAnalysisPanel)."""
        # This method is kept for backward compatibility but is no longer used
        # Button connections now call _start_esg_analysis('generate_reports')
        pass
        
    def _handle_reports_results(self, results: dict):
        """Handle report generation results."""
        self.reports = results
        
        # Update compliance scores
        gri = results['gri']
        tcfd = results['tcfd']
        sasb = results['sasb']
        
        self._update_kpi_card(
            self.lbl_gri_score,
            "GRI",
            f"{gri.compliance_score:.0f}%"
        )
        self._update_kpi_card(
            self.lbl_tcfd_score,
            "TCFD",
            f"{len(tcfd.disclosures)} disclosures"
        )
        self._update_kpi_card(
            self.lbl_sasb_score,
            "SASB",
            f"{sasb.compliance_score:.0f}%"
        )
        
        # Display first report
        self._display_report(self.combo_framework.currentText())
        
        self.btn_export_report.setEnabled(True)
        self.btn_generate_reports.setEnabled(True)
        self.status_label.setText("ESG reports generated")
        
    def _display_report(self, framework: str):
        """Display selected report."""
        if not self.reports:
            return
        
        if framework == 'GRI':
            report = self.reports['gri']
            text = f"GRI Standards Report\n"
            text += "=" * 60 + "\n\n"
            text += f"Reporting Period: {report.report_date}\n"
            text += f"Compliance Score: {report.compliance_score:.1f}%\n\n"
            text += f"Disclosures Provided: {len(report.disclosures)}\n"
            text += f"Gaps Identified: {len(report.gaps)}\n\n"
            
            text += "Key Disclosures:\n"
            for disclosure in report.disclosures[:10]:  # Show first 10
                text += f"  • {disclosure}\n"
                
        elif framework == 'TCFD':
            report = self.reports['tcfd']
            text = f"TCFD Report\n"
            text += "=" * 60 + "\n\n"
            text += f"Reporting Period: {report.report_date}\n"
            text += f"Compliance Score: {report.compliance_score:.1f}%\n\n"
            
            text += "Disclosures:\n"
            for disclosure in report.disclosures:
                text += f"  • {disclosure}\n"
                
        else:  # SASB
            report = self.reports['sasb']
            text = f"SASB Report (EM-MM)\n"
            text += "=" * 60 + "\n\n"
            text += f"Reporting Period: {report.report_date}\n"
            text += f"Compliance Score: {report.compliance_score:.1f}%\n\n"
            
            text += "Metrics:\n"
            for disclosure in report.disclosures[:15]:
                text += f"  • {disclosure}\n"
        
        self.text_report.setPlainText(text)
        
    def _export_reports(self):
        """Export reports to JSON."""
        if not self.reports:
            return
        
        from ..esg.governance import export_to_json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save ESG Reports", "esg_reports.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # Export all reports
                with open(file_path, 'w') as f:
                    import json
                    json.dump({
                        'gri': export_to_json(self.reports['gri']),
                        'tcfd': export_to_json(self.reports['tcfd']),
                        'sasb': export_to_json(self.reports['sasb'])
                    }, f, indent=2)
                
                QMessageBox.information(self, "Export Complete", f"Reports exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
    
    def _view_water_sankey(self):
        """Display interactive Sankey diagram for water balance."""
        if not self.water_data:
            QMessageBox.warning(self, "No Data", "No water balance data. Please simulate first.")
            return
        
        try:
            from ..visualization.sankey_diagram import WaterBalanceSankey
            from PyQt6.QtWidgets import QInputDialog
            
            # Ask user which backend to use
            backends = ['Interactive (Plotly)', 'Static (Matplotlib)']
            backend_choice, ok = QInputDialog.getItem(
                self,
                "Sankey Diagram Backend",
                "Select diagram type:",
                backends,
                0,
                False
            )
            
            if not ok:
                return
            
            backend = 'plotly' if 'Plotly' in backend_choice else 'matplotlib'
            
            # Build flows dictionary from water balance results
            flows = {}
            
            # Extract water balance metrics
            results = self.water_data.get('results', [])
            if results:
                # Use average flows across periods
                total_periods = len(results)
                
                # Sum up flows
                total_pit = sum(getattr(r, 'pit_water', 0) for r in results)
                total_fresh = sum(getattr(r, 'fresh_water', 0) for r in results)
                total_process = sum(getattr(r, 'water_use', 0) for r in results)
                total_recycled = sum(getattr(r, 'recycled', 0) for r in results)
                total_evap = sum(sum(getattr(r, 'evaporation', {}).values()) for r in results) if hasattr(results[0], 'evaporation') else total_process * 0.2
                total_discharge = sum(getattr(r, 'discharge', 0) for r in results) if hasattr(results[0], 'discharge') else total_process * 0.1
                
                # Build flows (convert to daily average)
                period_days = self.spin_period_days.value() if hasattr(self, 'spin_period_days') else 30
                
                if total_pit > 0:
                    flows['pit_to_process'] = (total_pit / total_periods) / period_days
                
                if total_fresh > 0:
                    flows['fresh_to_process'] = (total_fresh / total_periods) / period_days
                
                if total_process > 0:
                    # Estimate tailings input (assume 90% of process water)
                    flows['process_to_tailings'] = (total_process * 0.9 / total_periods) / period_days
                    flows['process_to_discharge'] = (total_discharge / total_periods) / period_days
                
                if total_recycled > 0:
                    flows['tailings_to_recycle'] = (total_recycled / total_periods) / period_days
                    flows['recycle_to_process'] = (total_recycled / total_periods) / period_days
                
                if total_evap > 0:
                    flows['tailings_to_evap'] = (total_evap / total_periods) / period_days
            
            if not flows:
                # Create example flows if no detailed data
                total_use = self.water_data.get('total_use', 1000)
                recycling_rate = self.water_data.get('recycling_rate', 0.6)
                
                flows = {
                    'pit_to_process': total_use * 0.3,
                    'fresh_to_process': total_use * 0.3,
                    'process_to_tailings': total_use * 0.8,
                    'tailings_to_recycle': total_use * recycling_rate,
                    'recycle_to_process': total_use * recycling_rate,
                    'tailings_to_evap': total_use * 0.15,
                    'process_to_discharge': total_use * 0.05
                }
            
            # Create Sankey diagram
            sankey = WaterBalanceSankey(backend=backend)
            
            self.status_label.setText("Generating Sankey diagram...")
            
            fig = sankey.create_water_balance_diagram(
                flows=flows,
                title='Water Balance - Mining Operation',
                units='m³/day',
                show_values=True,
                figsize=(16, 10)
            )
            
            if fig:
                if backend == 'plotly':
                    # Show Plotly figure in browser
                    fig.show()
                else:
                    # FIX 1: Non-blocking matplotlib - render to PNG and display in QLabel
                    import matplotlib.pyplot as plt
                    png_bytes = io.BytesIO()
                    fig.savefig(png_bytes, format='png', dpi=150, bbox_inches='tight')
                    plt.close(fig)  # Critical: close figure to free memory
                    png_bytes.seek(0)
                    
                    # Display in a QLabel with QPixmap
                    pixmap = QPixmap()
                    pixmap.loadFromData(png_bytes.read())
                    
                    # Create dialog to show the image
                    from PyQt6.QtWidgets import QDialog, QVBoxLayout, QScrollArea
                    dialog = QDialog(self)
                    dialog.setWindowTitle('Water Balance Sankey Diagram')
                    dialog.resize(1200, 800)
                    
                    layout = QVBoxLayout(dialog)
                    scroll = QScrollArea()
                    image_label = QLabel()
                    image_label.setPixmap(pixmap)
                    image_label.setScaledContents(False)
                    scroll.setWidget(image_label)
                    layout.addWidget(scroll)
                    
                    dialog.show()  # Non-modal
                
                self.status_label.setText("Sankey diagram displayed")
                
                # Ask if user wants to save
                from PyQt6.QtWidgets import QFileDialog
                save_option = QMessageBox.question(
                    self,
                    "Save Diagram",
                    "Would you like to save this Sankey diagram?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if save_option == QMessageBox.StandardButton.Yes:
                    if backend == 'plotly':
                        file_filter = "HTML Files (*.html);;PNG Images (*.png);;PDF Files (*.pdf);;All Files (*.*)"
                    else:
                        file_filter = "PNG Images (*.png);;PDF Files (*.pdf);;All Files (*.*)"
                    
                    file_path, _ = QFileDialog.getSaveFileName(
                        self,
                        "Save Sankey Diagram",
                        "",
                        file_filter
                    )
                    
                    if file_path:
                        sankey.save_diagram(file_path)
                        QMessageBox.information(
                            self,
                            "Diagram Saved",
                            f"Sankey diagram saved to:\n{file_path}"
                        )
                
                # Show efficiency metrics
                metrics = sankey.calculate_efficiency_metrics(flows)
                metrics_text = (
                    f"Water Efficiency Metrics:\n\n"
                    f"Recycling Rate: {metrics['recycling_rate']:.1f}%\n"
                    f"Fresh Water Intensity: {metrics['fresh_water_intensity']:.1f}%\n"
                    f"Loss Rate: {metrics['loss_rate']:.1f}%\n"
                    f"Total Inflow: {metrics['total_inflow']:.1f} m³/day\n"
                    f"Total Outflow: {metrics['total_outflow']:.1f} m³/day"
                )
                
                QMessageBox.information(self, "Water Efficiency", metrics_text)
            else:
                self.status_label.setText("Failed to create Sankey diagram")
                QMessageBox.warning(self, "Diagram Error", "Could not generate Sankey diagram.")
        
        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"Sankey diagram requires matplotlib (and optionally plotly):\n\n{str(e)}\n\n"
                f"Install with: pip install matplotlib plotly"
            )
        except Exception as e:
            logger.error(f"Error creating Sankey diagram: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Diagram Error",
                f"Error creating Sankey diagram:\n\n{str(e)}"
            )
            self.status_label.setText("Sankey diagram error")
                
    def _update_kpi_card(self, label: QLabel, title: str, value: str):
        """Update KPI card value."""
        current_style = label.styleSheet()
        label.setText(f"<b>{title}</b><br><span style='font-size:18pt'>{value}</span>")
        label.setStyleSheet(current_style)  # Preserve color
        
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def _start_esg_analysis(self, operation: str) -> None:
        """Start ESG analysis with specified operation."""
        self.current_operation = operation
        self.run_analysis()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI based on current operation."""
        if not self.schedule:
            raise ValueError("Please load a production schedule first.")
        
        if self.current_operation == "carbon_footprint":
            return {
                "operation": "carbon_footprint",
                "schedule": self.schedule,
            }
        elif self.current_operation == "water_balance":
            return {
                "operation": "water_balance",
                "schedule": self.schedule,
                "pit_capacity": getattr(self, 'spin_pit_capacity', None).value() if hasattr(self, 'spin_pit_capacity') else 50000.0,
                "pit_inflow": getattr(self, 'spin_pit_inflow', None).value() if hasattr(self, 'spin_pit_inflow') else 100.0,
                "tailings_capacity": getattr(self, 'spin_tailings_capacity', None).value() if hasattr(self, 'spin_tailings_capacity') else 500000.0,
                "tailings_area": getattr(self, 'spin_tailings_area', None).value() if hasattr(self, 'spin_tailings_area') else 50.0,
                "period_days": getattr(self, 'spin_period_days', None).value() if hasattr(self, 'spin_period_days') else 30,
            }
        elif self.current_operation == "waste_tracking":
            return {
                "operation": "waste_tracking",
                "schedule": self.schedule,
                "strip_ratio": getattr(self, 'spin_strip_ratio', None).value() if hasattr(self, 'spin_strip_ratio') else 3.0,
                "pag_percentage": getattr(self, 'spin_pag_percentage', None).value() / 100.0 if hasattr(self, 'spin_pag_percentage') else 0.05,
            }
        elif self.current_operation == "generate_reports":
            if not all([self.carbon_data, self.water_data, self.waste_data]):
                raise ValueError("Please calculate carbon, water, and waste metrics first.")
            return {
                "operation": "generate_reports",
                "carbon_data": self.carbon_data,
                "water_data": self.water_data,
                "waste_data": self.waste_data,
            }
        else:
            raise ValueError(f"Unknown ESG operation: {self.current_operation}")
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if not self.schedule:
            self.show_error("No Schedule", "Please load a production schedule first.")
            return False
        
        if self.current_operation == "generate_reports":
            if not all([self.carbon_data, self.water_data, self.waste_data]):
                self.show_error("Incomplete Data", "Please calculate carbon, water, and waste metrics first.")
                return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display ESG analysis results."""
        if payload is None:
            return
        
        if payload.get("error"):
            self.show_error("ESG Analysis Error", payload["error"])
            return
        
        operation = payload.get("operation") or self.current_operation
        
        if operation == "carbon_footprint":
            results = payload.get("results", {})
            # Convert to format expected by _handle_carbon_results
            carbon_results = {
                "total_co2e": results.get("total_co2e", 0.0),
                "intensity": results.get("intensity", 0.0),
                "summary": results.get("summary")
            }
            self._handle_carbon_results(carbon_results)
        elif operation == "water_balance":
            results = payload.get("results", {})
            self._handle_water_results(results)
        elif operation == "waste_tracking":
            results = payload.get("results", {})
            self._handle_waste_results(results)
        elif operation == "generate_reports":
            self._handle_reports_results(payload.get("results", {}))
    
    def _update_status(self, message: str):
        """Update status label."""
        if hasattr(self, 'status_label'):
            self.status_label.setText(message)
    
    def _update_progress(self, value: int):
        """Update progress bar value."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)
        
    def _handle_error(self, error_msg: str):
        """Handle worker errors."""
        self.show_error("Error", f"An error occurred:\n\n{error_msg}")
        if hasattr(self, 'status_label'):
            self.status_label.setText("Error occurred")
        
        # Re-enable buttons
        if hasattr(self, 'btn_calc_carbon'):
            self.btn_calc_carbon.setEnabled(True)
        if hasattr(self, 'btn_calc_water'):
            self.btn_calc_water.setEnabled(True)
        if hasattr(self, 'btn_calc_waste'):
            self.btn_calc_waste.setEnabled(True)
        if hasattr(self, 'btn_generate_reports'):
            self.btn_generate_reports.setEnabled(True)
    
    def closeEvent(self, event):
        """FIX 3: Stop timers/threads on close to prevent crashes."""
        try:
            # Stop any refresh timers
            if hasattr(self, "_refresh_timer") and self._refresh_timer:
                self._refresh_timer.stop()
                self._refresh_timer.deleteLater()
        except Exception as e:
            logger.warning(f"Error stopping refresh timer: {e}")
        
        try:
            # Cancel any running tasks via controller
            if self.controller and hasattr(self.controller, 'cancel_task'):
                self.controller.cancel_task(self.task_name)
        except Exception as e:
            logger.warning(f"Error cancelling tasks: {e}")
        
        super().closeEvent(event)
    
    def set_schedule(self, schedule: List):
        """Set schedule from external source (e.g., Underground panel)."""
        self.schedule = schedule
        self.lbl_schedule_status.setText(f"Loaded: {len(schedule)} periods")
        self.btn_calc_carbon.setEnabled(True)
        self.btn_calc_water.setEnabled(True)
        self.btn_calc_waste.setEnabled(True)
