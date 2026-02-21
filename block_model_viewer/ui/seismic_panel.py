"""
Seismic & Hazard Analysis Panel

Panel for loading seismic catalogues, building hazard volumes, and visualizing seismic data.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QFileDialog, QTabWidget, QWidget, QDateTimeEdit, QCheckBox
)
from PyQt6.QtCore import Qt, QDateTime

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class SeismicPanel(BaseAnalysisPanel):
    """
    Seismic & Hazard Analysis Panel.
    
    Provides:
    - Catalogue loading and filtering
    - Hazard volume construction
    - Statistics display (b-value, event rate)
    - Visualization controls
    """
    
    task_name = "seismic_hazard"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="seismic")
        
        self.catalog = None
        self.hazard_volume = None
        self._setup_ui()
        logger.info("Initialized Seismic & Hazard panel")
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Seismic & Hazard Analysis</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._create_catalogue_tab(), "Catalogue")
        tabs.addTab(self._create_hazard_tab(), "Hazard Volume")
        tabs.addTab(self._create_stats_tab(), "Statistics")
        layout.addWidget(tabs)
        
        self.tabs = tabs
    
    def _create_catalogue_tab(self) -> QWidget:
        """Create catalogue management tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Load catalogue
        load_group = QGroupBox("Load Catalogue")
        load_layout = QVBoxLayout(load_group)
        
        load_btn = QPushButton("Load Seismic Catalogue")
        load_btn.clicked.connect(self._load_catalogue)
        load_layout.addWidget(load_btn)
        
        self.catalogue_status_label = QLabel("No catalogue loaded")
        self.catalogue_status_label.setStyleSheet("color: orange;")
        load_layout.addWidget(self.catalogue_status_label)
        
        layout.addWidget(load_group)
        
        # Filters
        filter_group = QGroupBox("Filters")
        filter_layout = QFormLayout(filter_group)
        
        # Time window
        self.time_start_edit = QDateTimeEdit()
        self.time_start_edit.setCalendarPopup(True)
        self.time_start_edit.setDateTime(QDateTime.currentDateTime().addYears(-1))
        filter_layout.addRow("Start Time:", self.time_start_edit)
        
        self.time_end_edit = QDateTimeEdit()
        self.time_end_edit.setCalendarPopup(True)
        self.time_end_edit.setDateTime(QDateTime.currentDateTime())
        filter_layout.addRow("End Time:", self.time_end_edit)
        
        # Magnitude range
        self.mag_min_spinbox = QDoubleSpinBox()
        self.mag_min_spinbox.setRange(-2.0, 10.0)
        self.mag_min_spinbox.setValue(0.0)
        filter_layout.addRow("Min Magnitude:", self.mag_min_spinbox)
        
        self.mag_max_spinbox = QDoubleSpinBox()
        self.mag_max_spinbox.setRange(-2.0, 10.0)
        self.mag_max_spinbox.setValue(10.0)
        filter_layout.addRow("Max Magnitude:", self.mag_max_spinbox)
        
        # Spatial bounds (simplified)
        self.use_spatial_filter = QCheckBox("Apply Spatial Filter")
        filter_layout.addRow(self.use_spatial_filter)
        
        apply_filter_btn = QPushButton("Apply Filters")
        apply_filter_btn.clicked.connect(self._apply_filters)
        filter_layout.addRow(apply_filter_btn)
        
        layout.addWidget(filter_group)
        
        layout.addStretch()
        return widget
    
    def _create_hazard_tab(self) -> QWidget:
        """Create hazard volume construction tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Method selection
        method_group = QGroupBox("Hazard Method")
        method_layout = QFormLayout(method_group)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["distance", "kernel"])
        method_layout.addRow("Method:", self.method_combo)
        
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setRange(1.0, 1000.0)
        self.radius_spinbox.setValue(100.0)
        self.radius_spinbox.setSuffix(" m")
        method_layout.addRow("Influence Radius:", self.radius_spinbox)
        
        self.power_spinbox = QDoubleSpinBox()
        self.power_spinbox.setRange(0.5, 5.0)
        self.power_spinbox.setValue(2.0)
        method_layout.addRow("Distance Power:", self.power_spinbox)
        
        layout.addWidget(method_group)
        
        # Weighting options
        weight_group = QGroupBox("Weighting")
        weight_layout = QVBoxLayout(weight_group)
        
        self.magnitude_weight_checkbox = QCheckBox("Weight by Magnitude")
        self.magnitude_weight_checkbox.setChecked(True)
        weight_layout.addWidget(self.magnitude_weight_checkbox)
        
        self.energy_weight_checkbox = QCheckBox("Weight by Energy")
        weight_layout.addWidget(self.energy_weight_checkbox)
        
        self.time_decay_checkbox = QCheckBox("Apply Time Decay")
        weight_layout.addWidget(self.time_decay_checkbox)
        
        self.time_decay_spinbox = QDoubleSpinBox()
        self.time_decay_spinbox.setRange(1.0, 365.0)
        self.time_decay_spinbox.setValue(30.0)
        self.time_decay_spinbox.setSuffix(" days")
        self.time_decay_spinbox.setEnabled(False)
        self.time_decay_checkbox.stateChanged.connect(
            lambda state: self.time_decay_spinbox.setEnabled(state == Qt.CheckState.Checked.value)
        )
        weight_layout.addWidget(self.time_decay_spinbox)
        
        layout.addWidget(weight_group)
        
        # Run analysis
        run_btn = QPushButton("Build Hazard Volume")
        run_btn.clicked.connect(self._build_hazard_volume)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_stats_tab(self) -> QWidget:
        """Create statistics display tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)
        
        compute_stats_btn = QPushButton("Compute Statistics")
        compute_stats_btn.clicked.connect(self._compute_statistics)
        layout.addWidget(compute_stats_btn)
        
        return widget
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect hazard volume parameters."""
        params = {
            'method': self.method_combo.currentText(),
            'radius': self.radius_spinbox.value(),
            'power': self.power_spinbox.value(),
            'magnitude_weight': self.magnitude_weight_checkbox.isChecked(),
            'energy_weight': self.energy_weight_checkbox.isChecked(),
            'time_decay': self.time_decay_spinbox.value() if self.time_decay_checkbox.isChecked() else None
        }
        
        # Add catalogue if available
        if self.catalog:
            params['catalog'] = self.catalog
        
        return params
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if self.catalog is None or len(self.catalog.events) == 0:
            self.show_error("No Catalogue", "Please load a seismic catalogue first.")
            return False
        
        if not self.validate_block_model_loaded():
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle hazard volume results."""
        if payload.get('error'):
            self.show_error("Hazard Volume Error", payload['error'])
            return
        
        self.hazard_volume = payload.get('hazard_volume')
        
        # Update statistics
        self._compute_statistics()
        
        # Request visualization
        if 'visualization' in payload and self.controller:
            try:
                if hasattr(self.controller, 'renderer'):
                    renderer = self.controller.renderer
                    if hasattr(renderer, 'render_hazard_volume'):
                        renderer.render_hazard_volume(self.hazard_volume)
            except Exception as e:
                logger.warning(f"Failed to visualize hazard volume: {e}", exc_info=True)
        
        self.show_info("Success", "Hazard volume built successfully.")
    
    def _load_catalogue(self):
        """Load seismic catalogue from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Seismic Catalogue", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            from ..seismic.catalog import load_catalog
            
            self.catalog = load_catalog(Path(file_path))
            
            # Update status
            n_events = len(self.catalog.events)
            self.catalogue_status_label.setText(f"Loaded {n_events} events")
            self.catalogue_status_label.setStyleSheet("color: green;")
            
            # Update time range
            if self.catalog.events:
                time_start, time_end = self.catalog.get_time_range()
                self.time_start_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(time_start.timestamp())))
                self.time_end_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(time_end.timestamp())))
            
            self.show_info("Success", f"Loaded {n_events} seismic events.")
            
        except Exception as e:
            logger.error(f"Error loading catalogue: {e}", exc_info=True)
            self.show_error("Load Error", f"Failed to load catalogue: {e}")
    
    def _apply_filters(self):
        """Apply filters to catalogue."""
        if not self.catalog:
            self.show_warning("No Catalogue", "Please load a catalogue first.")
            return
        
        try:
            from ..seismic.catalog import filter_catalog
            
            # Get time window
            time_start = self.time_start_edit.dateTime().toPyDateTime()
            time_end = self.time_end_edit.dateTime().toPyDateTime()
            time_window = (time_start, time_end)
            
            # Get magnitude range
            mag_range = (self.mag_min_spinbox.value(), self.mag_max_spinbox.value())
            
            # Apply filters
            self.catalog = filter_catalog(
                self.catalog,
                time_window=time_window,
                mag_range=mag_range
            )
            
            # Update status
            n_events = len(self.catalog.events)
            self.catalogue_status_label.setText(f"Filtered: {n_events} events")
            
            self.show_info("Success", f"Applied filters: {n_events} events remaining.")
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}", exc_info=True)
            self.show_error("Filter Error", str(e))
    
    def _build_hazard_volume(self):
        """Build hazard volume."""
        self.run_analysis()
    
    def _compute_statistics(self):
        """Compute and display catalogue statistics."""
        if not self.catalog:
            self.stats_text.setPlainText("No catalogue loaded.")
            return
        
        try:
            from ..seismic.catalog import compute_b_value, compute_event_rate
            
            stats_lines = []
            stats_lines.append("=== Seismic Catalogue Statistics ===\n")
            
            # Basic info
            n_events = len(self.catalog.events)
            stats_lines.append(f"Total Events: {n_events}")
            
            if n_events > 0:
                # Time range
                time_start, time_end = self.catalog.get_time_range()
                stats_lines.append(f"Time Range: {time_start} to {time_end}")
                
                # Magnitude range
                magnitudes = self.catalog.get_magnitudes()
                stats_lines.append(f"Magnitude Range: {np.min(magnitudes):.2f} to {np.max(magnitudes):.2f}")
                stats_lines.append(f"Mean Magnitude: {np.mean(magnitudes):.2f}")
                
                # b-value
                b_value = compute_b_value(self.catalog)
                stats_lines.append(f"\nb-value: {b_value:.2f}")
                
                # Event rate
                event_rate_df = compute_event_rate(self.catalog)
                if not event_rate_df.empty:
                    total_days = (time_end - time_start).days
                    if total_days > 0:
                        avg_rate = n_events / total_days
                        stats_lines.append(f"Average Event Rate: {avg_rate:.2f} events/day")
            
            self.stats_text.setPlainText("\n".join(stats_lines))
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}", exc_info=True)
            self.stats_text.setPlainText(f"Error computing statistics: {e}")

