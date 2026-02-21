"""
DataRegistry Status Panel.

Provides a visual dashboard for monitoring the centralized DataRegistry:
- Current data availability
- Data flow connections
- Integrity checks
- Metadata information
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QTableWidget, QTableWidgetItem, QPushButton, QTextEdit,
    QScrollArea, QHeaderView, QTreeWidget, QTreeWidgetItem,
    QInputDialog, QFileDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont

from .base_analysis_panel import BaseAnalysisPanel
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class DataRegistryStatusPanel(BaseAnalysisPanel):
    """
    Panel for monitoring DataRegistry status and data flow.
    
    Displays:
    - Current registered data types and their status
    - Data flow graph showing connections between panels
    - Data integrity check results
    - Metadata for each registered data type
    """
    # PanelManager metadata
    PANEL_ID = "DataRegistryStatusPanel"
    PANEL_NAME = "DataRegistryStatus Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "data_registry_status"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="data_registry_status")
        self.setWindowTitle("Data Registry Status")
        
        # Connect to DataRegistry
        try:
            self.registry = self.get_registry()
            
            # Connect to all available signals to auto-refresh
            # Use hasattr to check if signal exists before connecting
            signals_to_connect = [
                'drillholeDataLoaded', 'blockModelLoaded', 'blockModelGenerated',
                'domainModelLoaded', 'contactSetLoaded', 'variogramResultsLoaded',
                'krigingResultsLoaded', 'sgsimResultsLoaded', 'simpleKrigingResultsLoaded',
                'cokrigingResultsLoaded', 'indicatorKrigingResultsLoaded', 
                'universalKrigingResultsLoaded', 'softKrigingResultsLoaded',
                'blockModelClassified', 'resourceCalculated', 'geometResultsLoaded',
                'pitOptimizationResultsLoaded', 'scheduleGenerated', 'irrResultsLoaded',
                'reconciliationResultsLoaded', 'experimentResultsLoaded',
                'drillholeDataCleared', 'blockModelCleared', 'domainModelCleared'
            ]
            
            connected_count = 0
            for signal_name in signals_to_connect:
                try:
                    signal = getattr(self.registry, signal_name, None)
                    if signal is not None and hasattr(signal, 'connect'):
                        signal.connect(self._on_data_changed)
                        connected_count += 1
                except Exception as e:
                    logger.debug(f"Could not connect to signal {signal_name}: {e}")
            
            logger.info(f"Connected to {connected_count}/{len(signals_to_connect)} DataRegistry signals")
            
        except Exception as e:
            logger.warning(f"Failed to initialize DataRegistry connection: {e}")
            self.registry = None
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.setInterval(2000)  # Refresh every 2 seconds
        
        self._setup_ui()
        self.refresh_status()
        logger.info("Initialized Data Registry Status panel")
    
    def _on_data_changed(self, *args):
        """Handle data change signals from DataRegistry."""
        self.refresh_status()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("<b>Data Registry Status Dashboard</b>")
        header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")
        layout.addWidget(header)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_status)
        controls_layout.addWidget(self.refresh_btn)
        
        self.integrity_btn = QPushButton("Run Integrity Check")
        self.integrity_btn.clicked.connect(self.run_integrity_check)
        controls_layout.addWidget(self.integrity_btn)
        
        self.clear_all_btn = QPushButton("Clear All Data")
        self.clear_all_btn.clicked.connect(self.clear_all_data)
        self.clear_all_btn.setStyleSheet("background-color: #f44336; color: white;")
        controls_layout.addWidget(self.clear_all_btn)
        
        # Add spacer instead of separator (layouts don't have addSeparator)
        controls_layout.addStretch()
        
        # Export buttons
        self.export_status_btn = QPushButton("Export Status Summary...")
        self.export_status_btn.clicked.connect(self.export_status_summary)
        self.export_status_btn.setStatusTip("Export status summary, integrity check, and data flow to JSON")
        controls_layout.addWidget(self.export_status_btn)
        
        self.export_flow_btn = QPushButton("Export Data Flow...")
        self.export_flow_btn.clicked.connect(self.export_data_flow)
        self.export_flow_btn.setStatusTip("Export data flow graph (JSON, CSV, or DOT format)")
        controls_layout.addWidget(self.export_flow_btn)
        
        # Add spacer instead of separator
        controls_layout.addStretch()
        
        self.auto_refresh_check = QPushButton("Auto-Refresh: OFF")
        self.auto_refresh_check.setCheckable(True)
        self.auto_refresh_check.clicked.connect(self._toggle_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_check)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(10)
        
        # Data Availability Table
        availability_group = QGroupBox("Data Availability")
        availability_layout = QVBoxLayout()
        
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels([
            "Data Type", "Status", "Source Panel", "Timestamp", "Details"
        ])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        availability_layout.addWidget(self.data_table)
        
        availability_group.setLayout(availability_layout)
        scroll_layout.addWidget(availability_group)
        
        # Data Flow Graph
        flow_group = QGroupBox("Data Flow Connections")
        flow_layout = QVBoxLayout()
        
        self.flow_tree = QTreeWidget()
        self.flow_tree.setHeaderLabel("Data Type → Consuming Panels")
        self.flow_tree.setAlternatingRowColors(True)
        flow_layout.addWidget(self.flow_tree)
        
        flow_group.setLayout(flow_layout)
        scroll_layout.addWidget(flow_group)
        
        # Integrity Check Results
        integrity_group = QGroupBox("Data Integrity")
        integrity_layout = QVBoxLayout()
        
        self.integrity_text = QTextEdit()
        self.integrity_text.setReadOnly(True)
        self.integrity_text.setMaximumHeight(150)
        integrity_layout.addWidget(self.integrity_text)
        
        integrity_group.setLayout(integrity_layout)
        scroll_layout.addWidget(integrity_group)
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def _toggle_auto_refresh(self, checked: bool):
        """Toggle auto-refresh timer."""
        if checked:
            self.refresh_timer.start()
            self.auto_refresh_check.setText("Auto-Refresh: ON")
            self.auto_refresh_check.setStyleSheet("background-color: #4CAF50; color: white;")
        else:
            self.refresh_timer.stop()
            self.auto_refresh_check.setText("Auto-Refresh: OFF")
            self.auto_refresh_check.setStyleSheet("")
    
    def refresh_status(self):
        """Refresh all status displays."""
        if not self.registry:
            return
        
        self._update_data_table()
        self._update_flow_tree()
    
    def _update_data_table(self):
        """Update the data availability table."""
        if not self.registry:
            return
        
        status_summary = self.registry.get_status_summary()
        
        # Define data types with their display names and getter methods
        data_types = [
            ("Drillhole Data", "drillhole_data", self.registry.get_drillhole_metadata),
            ("Block Model", "block_model", lambda: self.registry._block_model_metadata if hasattr(self.registry, '_block_model_metadata') else None),
            ("Domain Model", "domain_model", lambda: self.registry._domain_model_metadata if hasattr(self.registry, '_domain_model_metadata') else None),
            ("Contact Set", "contact_set", lambda: self.registry._contact_set_metadata if hasattr(self.registry, '_contact_set_metadata') else None),
            ("Variogram Results", "variogram_results", None),
            ("Kriging Results", "kriging_results", None),
            ("SGSIM Results", "sgsim_results", None),
            ("Simple Kriging", "simple_kriging_results", None),
            ("Co-Kriging", "cokriging_results", None),
            ("Indicator Kriging", "indicator_kriging_results", None),
            ("Universal Kriging", "universal_kriging_results", None),
            ("Soft Kriging", "soft_kriging_results", None),
            ("Classified Block Model", "classified_block_model", None),
            ("Resource Summary", "resource_summary", None),
            ("Geomet Results", "geomet_results", None),
            ("Geomet Ore Types", "geomet_ore_types", None),
            ("Pit Optimization", "pit_optimization_results", None),
            ("Schedule", "schedule", None),
            ("IRR Results", "irr_results", None),
            ("Reconciliation", "reconciliation_results", None),
            ("Haulage Evaluation", "haulage_evaluation", None),
            ("Experiment Results", "experiment_results", None),
        ]
        
        self.data_table.setRowCount(len(data_types))
        
        for row, (display_name, key, get_metadata) in enumerate(data_types):
            # Status
            is_available = status_summary.get(key, False)

            # Status item with theme-aware colors
            colors = get_theme_colors()
            status_item = QTableWidgetItem("✓ Available" if is_available else "✗ Not Available")
            status_item.setForeground(QColor(colors.SUCCESS) if is_available else QColor(colors.TEXT_DISABLED))
            self.data_table.setItem(row, 0, QTableWidgetItem(display_name))
            self.data_table.setItem(row, 1, status_item)
            
            # Metadata
            metadata = None
            if get_metadata:
                try:
                    metadata = get_metadata()
                except Exception:
                    pass
            
            if metadata:
                source = metadata.source_panel if hasattr(metadata, 'source_panel') else "Unknown"
                timestamp = metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(metadata, 'timestamp') and metadata.timestamp else "N/A"
                row_count = metadata.row_count if hasattr(metadata, 'row_count') and metadata.row_count else "N/A"
                details = f"{row_count} rows" if row_count != "N/A" else "N/A"
            else:
                source = "N/A"
                timestamp = "N/A"
                details = "N/A"
            
            self.data_table.setItem(row, 2, QTableWidgetItem(source))
            self.data_table.setItem(row, 3, QTableWidgetItem(timestamp))
            self.data_table.setItem(row, 4, QTableWidgetItem(details))
        
        self.data_table.resizeColumnsToContents()
    
    def _update_flow_tree(self):
        """Update the data flow tree widget."""
        if not self.registry:
            return
        
        self.flow_tree.clear()
        
        flow_graph = self.registry.get_data_flow_graph()
        
        for data_type, consuming_panels in flow_graph.items():
            # Create parent item for data type
            parent_item = QTreeWidgetItem(self.flow_tree, [data_type.replace('_', ' ').title()])
            parent_item.setExpanded(True)
            
            # Check if data is available - use theme colors
            colors = get_theme_colors()
            status_summary = self.registry.get_status_summary()
            is_available = status_summary.get(data_type, False)

            if is_available:
                parent_item.setForeground(0, QColor(colors.SUCCESS))
            else:
                parent_item.setForeground(0, QColor(colors.TEXT_DISABLED))

            # Add consuming panels as children
            for panel in consuming_panels:
                child_item = QTreeWidgetItem(parent_item, [panel])
                child_item.setForeground(0, QColor(colors.TEXT_SECONDARY))
        
        self.flow_tree.resizeColumnToContents(0)
    
    def run_integrity_check(self):
        """Run data integrity check and display results."""
        if not self.registry:
            self.integrity_text.setPlainText("Error: DataRegistry not available")
            return
        
        results = self.registry.check_data_integrity()
        
        output = "=" * 60 + "\n"
        output += "DATA INTEGRITY CHECK RESULTS\n"
        output += "=" * 60 + "\n\n"
        
        output += f"Checks Passed: {results['checks_passed']}\n"
        output += f"Checks Failed: {results['checks_failed']}\n\n"
        
        if results['warnings']:
            output += "WARNINGS:\n"
            output += "-" * 60 + "\n"
            for warning in results['warnings']:
                output += f"⚠ {warning}\n"
            output += "\n"
        
        if results['errors']:
            output += "ERRORS:\n"
            output += "-" * 60 + "\n"
            for error in results['errors']:
                output += f"✗ {error}\n"
            output += "\n"
        
        if not results['warnings'] and not results['errors']:
            output += "✓ All integrity checks passed!\n"
        
        output += "\n" + "=" * 60 + "\n"
        output += f"Check completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        self.integrity_text.setPlainText(output)
    
    def clear_all_data(self):
        """Clear all data from DataRegistry (with confirmation)."""
        from PyQt6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self,
            "Clear All Data",
            "Are you sure you want to clear ALL data from the DataRegistry?\n\n"
            "This will remove:\n"
            "- All drillhole data\n"
            "- All block models\n"
            "- All estimation results\n"
            "- All planning data\n"
            "- All other registered data\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.registry:
                self.registry.clear_all()
                self.refresh_status()
                logger.info("User cleared all DataRegistry data")
                QMessageBox.information(
                    self,
                    "Data Cleared",
                    "All data has been cleared from the DataRegistry."
                )
    
    def export_status_summary(self):
        """Export status summary to JSON file."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        
        if not self.registry:
            QMessageBox.warning(
                self,
                "Export Failed",
                "DataRegistry not available."
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Status Summary",
            str(Path.home() / "dataregistry_status.json"),
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.registry.export_status_summary(file_path):
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Status summary exported to:\n{file_path}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export status summary. Check logs for details."
                )
    
    def export_data_flow(self):
        """Export data flow graph to file."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        
        if not self.registry:
            QMessageBox.warning(
                self,
                "Export Failed",
                "DataRegistry not available."
            )
            return
        
        # Choose format
        format, ok = QInputDialog.getItem(
            self,
            "Export Format",
            "Select export format:",
            ["JSON", "CSV", "DOT (Graphviz)"],
            0,
            False
        )
        
        if not ok:
            return
        
        # Set default extension
        extensions = {
            "JSON": "json",
            "CSV": "csv",
            "DOT (Graphviz)": "dot"
        }
        ext = extensions.get(format, "json")
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Data Flow Graph ({format})",
            str(Path.home() / f"dataregistry_flow.{ext}"),
            f"{format} Files (*.{ext});;All Files (*)"
        )
        
        if file_path:
            format_lower = format.lower().split()[0]  # "DOT (Graphviz)" -> "dot"
            if self.registry.export_data_flow_graph(file_path, format_lower):
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Data flow graph exported to:\n{file_path}\n\nFormat: {format}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    "Failed to export data flow graph. Check logs for details."
                )

    def refresh_theme(self):
        """Refresh panel when theme changes."""
        # Call parent class refresh_theme (handles BaseAnalysisPanel styling)
        super().refresh_theme()

        # Refresh the data table and flow tree to update colors
        self._update_data_table()
        self._update_flow_tree()

