"""
Tactical Schedule Panel (STEP 30)

Monthly/quarterly pushback, bench/stope progression, and development scheduling.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
    QMessageBox, QTabWidget
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class TacticalSchedulePanel(BaseAnalysisPanel):
    """
    Panel for tactical scheduling (monthly/quarterly).
    """
    
    task_name = "tactical_pushback_schedule"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="tactical_schedule")
        self.strategic_schedule = None
        
        # Subscribe to strategic schedule from DataRegistry
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
        
        logger.info("Initialized Tactical Schedule Panel")
    


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
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Tactical Schedule Panel: Refine strategic schedule to monthly/quarterly pushback, bench, and development targets."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Tabs for different tactical scheduling types
        tabs = QTabWidget()
        
        # Pushback tab
        pushback_tab = self._create_pushback_tab()
        tabs.addTab(pushback_tab, "Pushback Schedule")
        
        # Bench tab
        bench_tab = self._create_bench_tab()
        tabs.addTab(bench_tab, "Bench Schedule")
        
        # Development tab
        dev_tab = self._create_development_tab()
        tabs.addTab(dev_tab, "Development Schedule")
        
        layout.addWidget(tabs)
        
        # Results
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_text)
    
    def _on_schedule_generated(self, schedule):
        """
        Automatically receive strategic schedule when it's generated.
        
        Args:
            schedule: Production schedule from DataRegistry (from Strategic Schedule)
        """
        logger.info("Tactical Schedule Panel received strategic schedule from DataRegistry")
        # Check if this is from strategic schedule (could filter by source_panel if available)
        self.strategic_schedule = schedule
        # Update UI to show that strategic schedule is available
        if hasattr(self, 'results_text'):
            self.results_text.setText(f"Strategic schedule received: {len(schedule) if isinstance(schedule, list) else 'N/A'} periods")
    
    def _publish_tactical_schedule(self, tactical_schedule):
        """
        Publish tactical schedule to DataRegistry.
        
        Args:
            tactical_schedule: Tactical schedule result
        """
        try:
            if hasattr(self, 'registry') and self.registry:
                self.registry.register_schedule(tactical_schedule, source_panel="TacticalSchedulePanel")
                logger.info("Published tactical schedule to DataRegistry")
        except Exception as e:
            logger.warning(f"Failed to publish tactical schedule to DataRegistry: {e}")
    
    def _create_pushback_tab(self) -> QWidget:
        """Create pushback scheduling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        config_group = QGroupBox("Pushback Configuration")
        config_layout = QFormLayout()
        
        self.max_pushbacks = QComboBox()
        self.max_pushbacks.addItems(["1", "2", "3", "4", "5"])
        self.max_pushbacks.setCurrentIndex(2)
        config_layout.addRow("Max Active Pushbacks:", self.max_pushbacks)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        run_btn = QPushButton("Run Pushback Schedule")
        run_btn.clicked.connect(self._on_run_pushback)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_bench_tab(self) -> QWidget:
        """Create bench scheduling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        config_group = QGroupBox("Bench Configuration")
        config_layout = QFormLayout()
        
        self.bench_height = QDoubleSpinBox()
        self.bench_height.setRange(5.0, 50.0)
        self.bench_height.setValue(15.0)
        self.bench_height.setSuffix(" m")
        config_layout.addRow("Bench Height:", self.bench_height)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        run_btn = QPushButton("Run Bench Schedule")
        run_btn.clicked.connect(self._on_run_bench)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def _create_development_tab(self) -> QWidget:
        """Create development scheduling tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        config_group = QGroupBox("Development Configuration")
        config_layout = QFormLayout()
        
        self.dev_rate = QDoubleSpinBox()
        self.dev_rate.setRange(0.0, 10000.0)
        self.dev_rate.setValue(500.0)
        self.dev_rate.setSuffix(" m/period")
        config_layout.addRow("Development Rate:", self.dev_rate)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        run_btn = QPushButton("Run Development Schedule")
        run_btn.clicked.connect(self._on_run_dev)
        layout.addWidget(run_btn)
        
        layout.addStretch()
        return widget
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def _on_run_pushback(self):
        """Run pushback schedule."""
        if not self.strategic_schedule:
            self.show_warning("No Strategic Schedule", "Please run strategic schedule first.")
            return
        
        params = {
            "strategic_schedule": self.strategic_schedule,
            "config": {
                "max_active_pushbacks": int(self.max_pushbacks.currentText())
            }
        }
        
        self.controller.run_tactical_pushback_schedule(params, self._on_complete)
    
    def _on_run_bench(self):
        """Run bench schedule."""
        params = {
            "block_model": self.controller.current_block_model if self.controller else None,
            "pushback_schedule": self.strategic_schedule,
            "bench_height": self.bench_height.value()
        }
        
        self.controller.run_tactical_bench_schedule(params, self._on_complete)
    
    def _on_run_dev(self):
        """Run development schedule."""
        params = {
            "tasks": [],  # Would get from UI
            "config": {
                "development_rate_m_per_period": self.dev_rate.value()
            }
        }
        
        self.controller.run_tactical_dev_schedule(params, self._on_complete)
    
    def _on_complete(self, result: Dict[str, Any]):
        """Handle completion."""
        self.results_text.append("Schedule complete.")
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

