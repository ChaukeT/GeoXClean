"""
Drillhole Metadata/Info Panel

Shows detailed information when a drillhole interval is selected.
Displays: Hole ID, From-To, Length, Survey azimuth/dip, Coordinates, Domain.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QGroupBox, QTableWidget, QTableWidgetItem,
    QPushButton, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

try:
    from .base_analysis_panel import BaseAnalysisPanel
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False

logger = logging.getLogger(__name__)


class DrillholeInfoPanel(BaseAnalysisPanel if BASE_AVAILABLE else QWidget):
    """
    Panel displaying drillhole interval metadata.
    
    Shows information when an interval is selected:
    - Hole ID
    - Depth From-To
    - Length
    - Survey azimuth/dip at that depth
    - 3D Coordinates (X, Y, Z)
    - Domain/Lithology
    - Assay values
    """
    
    # Signal emitted when user requests to focus on selected interval
    focus_requested = pyqtSignal(str, float, float)  # hole_id, depth_from, depth_to
    
    def __init__(self, parent=None):
        # Initialize data attributes before calling super().__init__
        self._current_interval: Optional[Dict[str, Any]] = None
        self.drillhole_data = None

        super().__init__(parent=parent, panel_id="drillhole_info" if BASE_AVAILABLE else None)

        # Connect to DataRegistry if available
        if BASE_AVAILABLE:
            self._connect_registry()

        self.setWindowTitle("Drillhole Information")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

        self._setup_ui()

    def _connect_registry(self):
        """Connect to DataRegistry for automatic drillhole data updates."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                # Connect to drillhole data signals
                self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                self.registry.drillholeDataCleared.connect(self._on_drillhole_data_cleared)

                # Load any existing drillhole data
                existing_data = self.registry.get_drillhole_data()
                if existing_data:
                    self._on_drillhole_data_loaded(existing_data)

                logger.info("Drillhole info panel connected to DataRegistry")
            else:
                logger.info("DataRegistry not available, drillhole info panel running standalone")
                self.registry = None
        except Exception as e:
            logger.warning(f"Failed to connect drillhole info panel to DataRegistry: {e}")
            self.registry = None

    def _on_drillhole_data_loaded(self, drillhole_data):
        """Handle new drillhole data loaded."""
        self.drillhole_data = drillhole_data
        logger.info("Drillhole info panel updated with new drillhole data")

    def _on_drillhole_data_cleared(self):
        """Handle drillhole data cleared."""
        self.drillhole_data = None
        logger.info("Drillhole info panel: drillhole data cleared")
    
    def _setup_ui(self):
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Drillhole Interval Information")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Basic Info Group
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        basic_group.setLayout(basic_layout)
        
        self.hole_id_label = QLabel("—")
        self.depth_from_label = QLabel("—")
        self.depth_to_label = QLabel("—")
        self.length_label = QLabel("—")
        
        basic_layout.addRow("Hole ID:", self.hole_id_label)
        basic_layout.addRow("Depth From:", self.depth_from_label)
        basic_layout.addRow("Depth To:", self.depth_to_label)
        basic_layout.addRow("Length:", self.length_label)
        
        layout.addWidget(basic_group)
        
        # Coordinates Group
        coord_group = QGroupBox("Coordinates")
        coord_layout = QFormLayout()
        coord_group.setLayout(coord_layout)
        
        self.x_label = QLabel("—")
        self.y_label = QLabel("—")
        self.z_label = QLabel("—")
        
        coord_layout.addRow("X:", self.x_label)
        coord_layout.addRow("Y:", self.y_label)
        coord_layout.addRow("Z:", self.z_label)
        
        layout.addWidget(coord_group)
        
        # Survey Group
        survey_group = QGroupBox("Survey")
        survey_layout = QFormLayout()
        survey_group.setLayout(survey_layout)
        
        self.azimuth_label = QLabel("—")
        self.dip_label = QLabel("—")
        
        survey_layout.addRow("Azimuth:", self.azimuth_label)
        survey_layout.addRow("Dip:", self.dip_label)
        
        layout.addWidget(survey_group)
        
        # Properties Group
        props_group = QGroupBox("Properties")
        props_layout = QFormLayout()
        props_group.setLayout(props_layout)
        
        self.lithology_label = QLabel("—")
        self.domain_label = QLabel("—")
        
        props_layout.addRow("Lithology:", self.lithology_label)
        props_layout.addRow("Domain:", self.domain_label)
        
        layout.addWidget(props_group)
        
        # Assay Values Table
        assay_group = QGroupBox("Assay Values")
        assay_layout = QVBoxLayout()
        assay_group.setLayout(assay_layout)
        
        self.assay_table = QTableWidget()
        self.assay_table.setColumnCount(2)
        self.assay_table.setHorizontalHeaderLabels(["Element", "Value"])
        self.assay_table.horizontalHeader().setStretchLastSection(True)
        self.assay_table.setMaximumHeight(150)
        assay_layout.addWidget(self.assay_table)
        
        layout.addWidget(assay_group)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        
        self.focus_btn = QPushButton("Focus on Interval")
        self.focus_btn.setEnabled(False)
        self.focus_btn.clicked.connect(self._on_focus_clicked)
        button_layout.addWidget(self.focus_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
        # Stretch
        layout.addStretch()
    
    def set_interval(self, interval_data: Dict[str, Any]):
        """
        Display information for a selected drillhole interval.
        
        Args:
            interval_data: Dictionary with keys:
                - hole_id: str
                - depth_from: float
                - depth_to: float
                - x, y, z: float (coordinates)
                - azimuth: Optional[float]
                - dip: Optional[float]
                - lith_code: Optional[str]
                - domain: Optional[str]
                - assay_values: Optional[Dict[str, float]]
        """
        self._current_interval = interval_data
        
        # Basic info
        self.hole_id_label.setText(str(interval_data.get("hole_id", "—")))
        self.depth_from_label.setText(f"{interval_data.get('depth_from', 0.0):.2f} m")
        self.depth_to_label.setText(f"{interval_data.get('depth_to', 0.0):.2f} m")
        
        length = abs(interval_data.get('depth_to', 0.0) - interval_data.get('depth_from', 0.0))
        self.length_label.setText(f"{length:.2f} m")
        
        # Coordinates
        self.x_label.setText(f"{interval_data.get('x', 0.0):.2f}")
        self.y_label.setText(f"{interval_data.get('y', 0.0):.2f}")
        self.z_label.setText(f"{interval_data.get('z', 0.0):.2f}")
        
        # Survey
        azimuth = interval_data.get('azimuth')
        dip = interval_data.get('dip')
        self.azimuth_label.setText(f"{azimuth:.1f}°" if azimuth is not None else "—")
        self.dip_label.setText(f"{dip:.1f}°" if dip is not None else "—")
        
        # Properties - use label mapping for lithology if available
        lith_code = interval_data.get('lith_code', '—')
        if lith_code != '—' and hasattr(self, 'registry') and self.registry:
            try:
                lith_label = self.registry.get_category_label("drillholes.lithology", str(lith_code))
                self.lithology_label.setText(lith_label)
            except Exception:
                self.lithology_label.setText(str(lith_code))
        else:
            self.lithology_label.setText(str(lith_code))
        self.domain_label.setText(str(interval_data.get('domain', '—')))
        
        # Assay values
        assay_values = interval_data.get('assay_values', {})
        self.assay_table.setRowCount(len(assay_values))
        for row, (element, value) in enumerate(assay_values.items()):
            self.assay_table.setItem(row, 0, QTableWidgetItem(str(element)))
            self.assay_table.setItem(row, 1, QTableWidgetItem(f"{value:.4f}"))
        
        self.focus_btn.setEnabled(True)
        
        logger.debug(f"Updated drillhole info panel for interval: {interval_data.get('hole_id')}")
    
    def clear(self):
        """Clear all displayed information."""
        self._current_interval = None
        
        self.hole_id_label.setText("—")
        self.depth_from_label.setText("—")
        self.depth_to_label.setText("—")
        self.length_label.setText("—")
        self.x_label.setText("—")
        self.y_label.setText("—")
        self.z_label.setText("—")
        self.azimuth_label.setText("—")
        self.dip_label.setText("—")
        self.lithology_label.setText("—")
        self.domain_label.setText("—")
        self.assay_table.setRowCount(0)
        
        self.focus_btn.setEnabled(False)
    
    def _on_focus_clicked(self):
        """Handle focus button click."""
        if self._current_interval:
            hole_id = self._current_interval.get('hole_id')
            depth_from = self._current_interval.get('depth_from', 0.0)
            depth_to = self._current_interval.get('depth_to', 0.0)
            self.focus_requested.emit(hole_id, depth_from, depth_to)

