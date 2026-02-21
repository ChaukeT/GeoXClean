"""
Production Dashboard Panel (STEP 36)

Joint dashboard aligning NPVS schedules, haulage capacity, and reconciliation results.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QTextEdit, QMessageBox
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import pyqtSlot

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class ProductionDashboardPanel(BaseAnalysisPanel):
    """
    Production Dashboard Panel.
    
    Aligns planned (NPVS), haulage capacity, and actual (reconciliation) production data.
    """
    # PanelManager metadata
    PANEL_ID = "ProductionDashboardPanel"
    PANEL_NAME = "ProductionDashboard Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    PANEL_ID = "ProductionDashboardPanel"  # STEP 40
    task_name = "production_align"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="production_dashboard")
        self.current_result = None
        
        # Subscribe to data updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            self.registry.reconciliationResultsLoaded.connect(self._on_reconciliation_results_loaded)
            self.registry.haulageEvaluationLoaded.connect(self._on_haulage_evaluation_loaded)
            
            # Load existing data if available
            existing_schedule = self.registry.get_schedule()
            if existing_schedule:
                self._on_schedule_generated(existing_schedule)
            
            existing_reconciliation = self.registry.get_reconciliation_results()
            if existing_reconciliation:
                self._on_reconciliation_results_loaded(existing_reconciliation)
            
            existing_haulage = self.registry.get_haulage_evaluation()
            if existing_haulage:
                self._on_haulage_evaluation_loaded(existing_haulage)
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
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Production Dashboard: Align planned schedules, haulage capacity, "
            "and actual production (reconciliation) to identify variances and bottlenecks."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Data source selection
        layout.addWidget(self._build_data_source_section())
        
        # Summary KPIs
        layout.addWidget(self._build_kpi_section())
        
        # Main tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_period_table(), "Per-Period Metrics")
        tabs.addTab(self._build_tonnes_variance_table(), "Tonnes Variance")
        tabs.addTab(self._build_grade_variance_table(), "Grade Variance")
        tabs.addTab(self._build_haulage_table(), "Haulage vs Plan")
        layout.addWidget(tabs)
        
        # Export button
        export_btn = QPushButton("Export Period Metrics (CSV)")
        export_btn.clicked.connect(self._on_export)
        layout.addWidget(export_btn)
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Production Dashboard received schedule from DataRegistry")
        self.schedule = schedule
        # Update UI with schedule data
        # Implementation depends on schedule format
    
    def _on_reconciliation_results_loaded(self, reconciliation_results):
        """
        Automatically receive reconciliation results when they're loaded.
        
        Args:
            reconciliation_results: Reconciliation results from DataRegistry
        """
        logger.info("Production Dashboard received reconciliation results from DataRegistry")
        self.reconciliation_results = reconciliation_results
        # Update UI with reconciliation data
    
    def _on_haulage_evaluation_loaded(self, haulage_evaluation):
        """
        Automatically receive haulage evaluation when it's loaded.
        
        Args:
            haulage_evaluation: Haulage evaluation from DataRegistry
        """
        logger.info("Production Dashboard received haulage evaluation from DataRegistry")
        self.haulage_evaluation = haulage_evaluation
        # Update UI with haulage data
    
    def _build_data_source_section(self) -> QWidget:
        """Build data source selection section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Data Sources")
        form = QFormLayout(group)
        
        self.schedule_source = QComboBox()
        self.schedule_source.addItems([
            "Use latest NPVS schedule",
            "Use current strategic schedule",
            "Select from scenario..."
        ])
        form.addRow("Schedule Source:", self.schedule_source)
        
        self.haulage_source = QComboBox()
        self.haulage_source.addItems([
            "Use last haulage evaluation",
            "Run haulage evaluation now..."
        ])
        form.addRow("Haulage Source:", self.haulage_source)
        
        self.recon_source = QComboBox()
        self.recon_source.addItems([
            "Use latest reconciliation run",
            "Select reconciliation dataset..."
        ])
        form.addRow("Reconciliation Source:", self.recon_source)
        
        align_btn = QPushButton("Align Production Data")
        align_btn.clicked.connect(self._on_align_data)
        form.addRow(align_btn)
        
        layout.addWidget(group)
        return widget
    
    def _build_kpi_section(self) -> QWidget:
        """Build summary KPI section."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        group = QGroupBox("Summary KPIs")
        kpi_layout = QVBoxLayout(group)
        
        self.kpi_text = QTextEdit()
        self.kpi_text.setReadOnly(True)
        self.kpi_text.setMaximumHeight(150)
        kpi_layout.addWidget(self.kpi_text)
        
        layout.addWidget(group)
        return widget
    
    def _build_period_table(self) -> QWidget:
        """Build per-period metrics table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.period_table = QTableWidget()
        self.period_table.setColumnCount(13)
        self.period_table.setHorizontalHeaderLabels([
            "Period", "Planned Mined (t)", "Actual Mined (t)",
            "Planned Mill (t)", "Actual Mill (t)",
            "Planned Grade", "Mill Grade",
            "Haulage Util (%)", "Haulage Shortfall (t)",
            "Δ Mined (%)", "Δ Mill (%)",
            "Grade Bias Mine", "Grade Bias Mill"
        ])
        self.period_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.period_table)
        
        return widget
    
    def _build_tonnes_variance_table(self) -> QWidget:
        """Build tonnes variance table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.tonnes_table = QTableWidget()
        self.tonnes_table.setColumnCount(6)
        self.tonnes_table.setHorizontalHeaderLabels([
            "Period", "Planned Mined (t)", "Actual Mined (t)",
            "Δ Mined (t)", "Δ Mined (%)", "Status"
        ])
        layout.addWidget(self.tonnes_table)
        
        return widget
    
    def _build_grade_variance_table(self) -> QWidget:
        """Build grade variance table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.grade_table = QTableWidget()
        self.grade_table.setColumnCount(5)
        self.grade_table.setHorizontalHeaderLabels([
            "Period", "Element", "Planned Grade", "Actual Grade", "Bias"
        ])
        layout.addWidget(self.grade_table)
        
        return widget
    
    def _build_haulage_table(self) -> QWidget:
        """Build haulage vs plan table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.haulage_table = QTableWidget()
        self.haulage_table.setColumnCount(5)
        self.haulage_table.setHorizontalHeaderLabels([
            "Period", "Planned (t)", "Hauled (t)",
            "Utilisation (%)", "Shortfall (t)"
        ])
        layout.addWidget(self.haulage_table)
        
        return widget
    
    @pyqtSlot()
    def _on_align_data(self):
        """Handle align data button click."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        # Build config from UI selections
        config = {
            "schedule_result": None,  # Would load from selected source
            "haulage_eval_result": None,  # Would load from selected source
            "recon_result": None  # Would load from selected source
        }
        
        # For now, try to get latest results from controller
        # In practice, would load from scenario store or latest runs
        
        self.show_progress("Aligning production data...")
        
        try:
            self.controller.align_production(config, self._on_alignment_done)
        except Exception as e:
            logger.error(f"Failed to align production: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("Alignment Failed", f"Failed to align production data:\n{e}")
    
    @pyqtSlot(dict)
    def _on_alignment_done(self, result: dict):
        """Handle alignment completion."""
        self.hide_progress()
        
        # Handle result payload structure
        if isinstance(result, dict) and "result" in result:
            actual_result = result["result"]
        else:
            actual_result = result
        
        if actual_result.get("error"):
            self.show_error("Alignment Error", actual_result["error"])
            return
        
        self.current_result = actual_result
        self._update_display()
    
    def _update_display(self):
        """Update all displays with current result."""
        if not self.current_result:
            return
        
        # Update KPIs
        kpis = self.current_result.get("overall_kpis", {})
        kpi_text = f"""
NPV: ${kpis.get('npv', 0):,.0f}
Total Planned Mined: {kpis.get('total_planned_mined_t', 0):,.0f} t
Total Actual Mined: {kpis.get('total_mined_actual_t', 0):,.0f} t
Mined Variance: {kpis.get('mined_variance_pct', 0):.1f}%
Total Planned Mill: {kpis.get('total_planned_mill_t', 0):,.0f} t
Total Actual Mill: {kpis.get('total_mill_actual_t', 0):,.0f} t
Mill Variance: {kpis.get('mill_variance_pct', 0):.1f}%
Avg Haulage Utilisation: {kpis.get('avg_haulage_utilisation', 0):.1f}%
Periods with Haulage Shortfall: {kpis.get('periods_with_haulage_shortfall', 0)} / {kpis.get('total_periods', 0)}
        """
        self.kpi_text.setPlainText(kpi_text.strip())
        
        # Update period table
        periods = self.current_result.get("periods", [])
        self._fill_period_table(periods)
        self._fill_tonnes_table(periods)
        self._fill_haulage_table(periods)
    
    def _fill_period_table(self, periods: list):
        """Fill per-period metrics table."""
        self.period_table.setRowCount(len(periods))
        
        for r, period in enumerate(periods):
            if isinstance(period, dict):
                p = period
            else:
                # Convert dataclass to dict
                p = {
                    'period_id': getattr(period, 'period_id', ''),
                    'planned_mined_t': getattr(period, 'planned_mined_t', 0.0),
                    'mined_actual_t': getattr(period, 'mined_actual_t', 0.0),
                    'planned_plant_t': getattr(period, 'planned_plant_t', 0.0),
                    'mill_actual_t': getattr(period, 'mill_actual_t', 0.0),
                    'haulage_utilisation': getattr(period, 'haulage_utilisation', 0.0),
                    'haulage_shortfall_t': getattr(period, 'haulage_shortfall_t', 0.0),
                    'delta_mined_t': getattr(period, 'delta_mined_t', 0.0),
                    'delta_mill_t': getattr(period, 'delta_mill_t', 0.0),
                }
            
            planned_mined = p.get('planned_mined_t', 0.0)
            actual_mined = p.get('mined_actual_t', 0.0)
            planned_mill = p.get('planned_plant_t', 0.0)
            actual_mill = p.get('mill_actual_t', 0.0)
            
            delta_mined_pct = (p.get('delta_mined_t', 0.0) / planned_mined * 100) if planned_mined > 0 else 0.0
            delta_mill_pct = (p.get('delta_mill_t', 0.0) / planned_mill * 100) if planned_mill > 0 else 0.0
            
            self.period_table.setItem(r, 0, QTableWidgetItem(str(p.get('period_id', ''))))
            self.period_table.setItem(r, 1, QTableWidgetItem(f"{planned_mined:,.0f}"))
            self.period_table.setItem(r, 2, QTableWidgetItem(f"{actual_mined:,.0f}"))
            self.period_table.setItem(r, 3, QTableWidgetItem(f"{planned_mill:,.0f}"))
            self.period_table.setItem(r, 4, QTableWidgetItem(f"{actual_mill:,.0f}"))
            self.period_table.setItem(r, 5, QTableWidgetItem(""))  # Planned grade
            self.period_table.setItem(r, 6, QTableWidgetItem(""))  # Mill grade
            self.period_table.setItem(r, 7, QTableWidgetItem(f"{p.get('haulage_utilisation', 0.0):.1f}"))
            self.period_table.setItem(r, 8, QTableWidgetItem(f"{p.get('haulage_shortfall_t', 0.0):,.0f}"))
            self.period_table.setItem(r, 9, QTableWidgetItem(f"{delta_mined_pct:.1f}"))
            self.period_table.setItem(r, 10, QTableWidgetItem(f"{delta_mill_pct:.1f}"))
            self.period_table.setItem(r, 11, QTableWidgetItem(""))  # Grade bias mine
            self.period_table.setItem(r, 12, QTableWidgetItem(""))  # Grade bias mill
        
        self.period_table.resizeColumnsToContents()
    
    def _fill_tonnes_table(self, periods: list):
        """Fill tonnes variance table."""
        self.tonnes_table.setRowCount(len(periods))
        
        for r, period in enumerate(periods):
            if isinstance(period, dict):
                p = period
            else:
                p = {
                    'period_id': getattr(period, 'period_id', ''),
                    'planned_mined_t': getattr(period, 'planned_mined_t', 0.0),
                    'mined_actual_t': getattr(period, 'mined_actual_t', 0.0),
                    'delta_mined_t': getattr(period, 'delta_mined_t', 0.0),
                }
            
            planned = p.get('planned_mined_t', 0.0)
            actual = p.get('mined_actual_t', 0.0)
            delta = p.get('delta_mined_t', 0.0)
            delta_pct = (delta / planned * 100) if planned > 0 else 0.0
            
            status = "On Target" if abs(delta_pct) < 5 else ("Over" if delta > 0 else "Under")
            
            self.tonnes_table.setItem(r, 0, QTableWidgetItem(str(p.get('period_id', ''))))
            self.tonnes_table.setItem(r, 1, QTableWidgetItem(f"{planned:,.0f}"))
            self.tonnes_table.setItem(r, 2, QTableWidgetItem(f"{actual:,.0f}"))
            self.tonnes_table.setItem(r, 3, QTableWidgetItem(f"{delta:,.0f}"))
            self.tonnes_table.setItem(r, 4, QTableWidgetItem(f"{delta_pct:.1f}%"))
            self.tonnes_table.setItem(r, 5, QTableWidgetItem(status))
        
        self.tonnes_table.resizeColumnsToContents()
    
    def _fill_haulage_table(self, periods: list):
        """Fill haulage vs plan table."""
        self.haulage_table.setRowCount(len(periods))
        
        for r, period in enumerate(periods):
            if isinstance(period, dict):
                p = period
            else:
                p = {
                    'period_id': getattr(period, 'period_id', ''),
                    'planned_mined_t': getattr(period, 'planned_mined_t', 0.0),
                    'hauled_t': getattr(period, 'hauled_t', 0.0),
                    'haulage_utilisation': getattr(period, 'haulage_utilisation', 0.0),
                    'haulage_shortfall_t': getattr(period, 'haulage_shortfall_t', 0.0),
                }
            
            self.haulage_table.setItem(r, 0, QTableWidgetItem(str(p.get('period_id', ''))))
            self.haulage_table.setItem(r, 1, QTableWidgetItem(f"{p.get('planned_mined_t', 0.0):,.0f}"))
            self.haulage_table.setItem(r, 2, QTableWidgetItem(f"{p.get('hauled_t', 0.0):,.0f}"))
            self.haulage_table.setItem(r, 3, QTableWidgetItem(f"{p.get('haulage_utilisation', 0.0):.1f}"))
            self.haulage_table.setItem(r, 4, QTableWidgetItem(f"{p.get('haulage_shortfall_t', 0.0):,.0f}"))
        
        self.haulage_table.resizeColumnsToContents()
    
    @pyqtSlot()
    def _on_export(self):
        """Export period metrics to CSV."""
        if not self.current_result:
            self.show_warning("No Data", "No aligned data to export.")
            return
        
        try:
            from ..utils.data_bridge import aligned_dashboard_result_to_dataframe
            
            df = aligned_dashboard_result_to_dataframe(self.current_result)
            if df.empty:
                self.show_warning("Export Failed", "No data to export.")
                return
            
            # Would show file dialog here
            self.show_info("Export", "Export functionality ready (file dialog to be implemented)")
        
        except Exception as e:
            logger.error(f"Failed to export: {e}", exc_info=True)
            self.show_error("Export Failed", f"Failed to export data:\n{e}")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        self._on_alignment_done(payload)

