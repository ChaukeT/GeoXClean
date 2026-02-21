"""
Geotech Summary Panel - Dashboard for geotechnical state (STEP 27).
"""

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QGroupBox, QLabel,
    QTableWidget, QTableWidgetItem, QTextEdit
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class GeotechSummaryPanel(BaseAnalysisPanel):
    """Panel for geotechnical summary dashboard."""
    # PanelManager metadata
    PANEL_ID = "GeotechSummaryPanel"
    PANEL_NAME = "GeotechSummary Panel"
    PANEL_CATEGORY = PanelCategory.GEOTECH
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "geotech_summary"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="geotech_summary")
        self.setWindowTitle("Geotechnical Summary")
        # Sizing is handled by BaseAnalysisPanel._setup_panel_sizing()
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        title = QLabel("Geotechnical Summary Dashboard")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Summary tables
        materials_group = QGroupBox("Material Properties by Domain")
        materials_layout = QVBoxLayout()
        self.materials_table = QTableWidget()
        self.materials_table.setColumnCount(4)
        self.materials_table.setHorizontalHeaderLabels(["Domain", "Material", "Friction Angle", "Cohesion"])
        materials_layout.addWidget(self.materials_table)
        materials_group.setLayout(materials_layout)
        layout.addWidget(materials_group)
        
        stability_group = QGroupBox("Slope Stability Results")
        stability_layout = QVBoxLayout()
        self.stability_table = QTableWidget()
        self.stability_table.setColumnCount(4)
        self.stability_table.setHorizontalHeaderLabels(["Sector", "Domain", "FOS", "Status"])
        stability_layout.addWidget(self.stability_table)
        stability_group.setLayout(stability_layout)
        layout.addWidget(stability_group)
        
        # Summary text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        
        self.summary_text.append("Geotechnical Summary Dashboard")
        self.summary_text.append("This panel provides an overview of:")
        self.summary_text.append("- Rock mass properties per domain")
        self.summary_text.append("- Material properties per domain")
        self.summary_text.append("- Structural sets influencing slopes")
        self.summary_text.append("- Slope stability results per sector")
        self.summary_text.append("- Bench design rules per domain")

