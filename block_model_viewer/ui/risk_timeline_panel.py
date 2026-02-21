"""
Risk Timeline Panel

Panel for visualizing risk vs time for one or more schedules.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QWidget
)
from PyQt6.QtCore import Qt

try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    Figure = None

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class RiskTimelinePanel(BaseAnalysisPanel):
    """
    Risk Timeline Visualization Panel.
    
    Provides:
    - Risk vs period plots
    - Overlay plots for multiple schedules
    - Threshold highlighting
    - Exposure summaries
    """
    
    task_name = "schedule_risk_timeline"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="risk_timeline")
        
        self.current_profiles = []
        self.current_comparison = None
        self.schedule = None
        
        # Subscribe to schedule from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.scheduleGenerated.connect(self._on_schedule_generated)
            
            # Load existing schedule if available
            existing_schedule = self.registry.get_schedule()
            if existing_schedule:
                self._on_schedule_generated(existing_schedule)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self.setup_ui()
        logger.info("Initialized Risk Timeline panel")
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry
        """
        logger.info("Risk Timeline Panel received schedule from DataRegistry")
        self.schedule = schedule
        # Update UI to reflect schedule is available
        if hasattr(self, 'metric_combo'):
            # Enable analysis if schedule is available
            pass
    
    def setup_ui(self):
        """Setup the UI layout."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("<b>Risk Timeline Visualization</b>")
        title_label.setStyleSheet("font-size: 14px; padding: 5px;")
        layout.addWidget(title_label)
        
        if not MATPLOTLIB_AVAILABLE:
            error_label = QLabel("Matplotlib not available. Install matplotlib to use timeline visualization.")
            error_label.setStyleSheet("color: red; font-weight: bold;")
            layout.addWidget(error_label)
            return
        
        # Plot settings
        settings_group = QGroupBox("Plot Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "combined_risk_score",
            "seismic_hazard_index",
            "rockburst_risk_index",
            "slope_risk_index"
        ])
        settings_layout.addRow("Metric:", self.metric_combo)
        
        self.use_time_combo = QComboBox()
        self.use_time_combo.addItems(["Period Index", "Time (Days)"])
        settings_layout.addRow("X-Axis:", self.use_time_combo)
        
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 1.0)
        self.threshold_spinbox.setValue(0.5)
        self.threshold_spinbox.setSingleStep(0.1)
        settings_layout.addRow("Risk Threshold:", self.threshold_spinbox)
        
        layout.addWidget(settings_group)
        
        # Plot button
        plot_btn = QPushButton("Plot Risk Timeline")
        plot_btn.clicked.connect(self._plot_timeline)
        layout.addWidget(plot_btn)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Summary text
        self.summary_text = QLabel()
        self.summary_text.setWordWrap(True)
        layout.addWidget(self.summary_text)
        
        layout.addStretch()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect timeline parameters."""
        return {
            'metric': self.metric_combo.currentText(),
            'use_time': self.use_time_combo.currentIndex() == 1,
            'threshold': self.threshold_spinbox.value()
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.current_profiles and not self.current_comparison:
            self.show_warning("No Data", "Please load a risk profile first.")
            return False
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle timeline results."""
        if payload.get('error'):
            self.show_error("Timeline Error", payload['error'])
            return
        
        time_series = payload.get('time_series', [])
        exposure_stats = payload.get('exposure_stats')
        
        if time_series:
            self._plot_time_series(time_series, exposure_stats)
        
        self.show_info("Success", "Risk timeline plotted.")
    
    def _plot_timeline(self):
        """Plot risk timeline."""
        if not self.current_profiles:
            self.show_warning("No Profile", "Please build a risk profile first.")
            return
        
        # Use first profile for now
        profile = self.current_profiles[0]
        params = self.gather_parameters()
        params['profile'] = profile
        
        if self.controller:
            self.controller.run_task("schedule_risk_timeline", params, self.handle_results)
    
    def _plot_time_series(self, time_series: List[tuple], exposure_stats: Optional[Dict[str, Any]] = None):
        """Plot time series data."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if not time_series:
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        times, values = zip(*time_series)
        
        # Plot main line
        ax.plot(times, values, 'b-', linewidth=2, label='Risk Score')
        
        # Add threshold line if provided
        if exposure_stats and 'threshold' in exposure_stats:
            threshold = exposure_stats['threshold']
            ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
            
            # Highlight periods above threshold
            above_threshold = [v for v in values if v >= threshold]
            if above_threshold:
                ax.fill_between(times, threshold, values, where=[v >= threshold for v in values],
                              alpha=0.3, color='red', label='High Risk')
        
        ax.set_xlabel('Period Index' if not self.use_time_combo.currentIndex() == 1 else 'Time (Days)')
        ax.set_ylabel('Risk Score')
        ax.set_title('Risk Timeline')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Update summary text
        if exposure_stats:
            summary_lines = [
                f"Periods above threshold: {exposure_stats.get('total_periods_above_threshold', 0)}",
                f"Percentage: {exposure_stats.get('pct_periods_above_threshold', 0):.1f}%",
                f"Tonnage at high risk: {exposure_stats.get('total_tonnage_high_risk', 0):,.0f} t",
                f"Percentage: {exposure_stats.get('pct_tonnage_high_risk', 0):.1f}%"
            ]
            self.summary_text.setText("\n".join(summary_lines))
        
        self.canvas.draw()

