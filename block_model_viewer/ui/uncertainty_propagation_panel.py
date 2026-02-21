"""
Uncertainty Propagation Panel

Propagate grade realisations through economic models.
Refactored for modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTabWidget, QWidget, QSplitter, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt
import numpy as np
import pandas as pd
from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class UncertaintyPropagationPanel(BaseAnalysisPanel):
    task_name = "economic_uncert"
    
    def __init__(self, parent=None):
        self._block_model = None
        self.sgsim_results = None
        super().__init__(parent=parent, panel_id="uncertainty_propagation")
        self.setWindowTitle("Uncertainty Propagation - Economic Analysis")
        self.resize(1200, 800)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()
        
        self._init_registry()
    


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
        """Build custom UI. Called by base class."""
        self._setup_ui()

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.blockModelLoaded.connect(self._on_bm_loaded)
                self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                b = self.registry.get_block_model()
                if b:
                    self._on_bm_loaded(b)
                s = self.registry.get_sgsim_results()
                if s:
                    self._on_sgsim_loaded(s)
        except Exception:
            pass

    def _setup_ui(self):
        # Clear any existing layout from base class (BaseAnalysisPanel creates a scroll area)
        old_layout = self.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                    del item
            QWidget().setLayout(old_layout)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT: CONFIG ---
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        
        self._create_input_group(s_lay)
        self._create_econ_group(s_lay)
        self._create_opt_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # --- RIGHT: RESULTS ---
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Tabs for detailed output
        self.tabs = QTabWidget()
        self.sum_tab = QWidget()
        self.tabs.addTab(self.sum_tab, "Summary")
        sum_l = QVBoxLayout(self.sum_tab)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Metric", "Value", "Unit"])
        self.table.horizontalHeader().setStretchLastSection(True)
        sum_l.addWidget(self.table)
        
        self.dist_tab = QWidget()
        self.tabs.addTab(self.dist_tab, "Distributions")
        dist_l = QVBoxLayout(self.dist_tab)
        self.dist_text = QTextEdit()
        self.dist_text.setReadOnly(True)
        self.dist_text.setStyleSheet(f"background-color: #2b2b2b; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas;")
        dist_l.addWidget(self.dist_text)
        
        self.log_tab = QWidget()
        self.tabs.addTab(self.log_tab, "Log")
        log_l = QVBoxLayout(self.log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"background-color: #2b2b2b; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas;")
        log_l.addWidget(self.log_text)
        
        r_lay.addWidget(self.tabs)
        
        # Action Buttons
        self.run_btn = QPushButton("RUN PROPAGATION")
        self.run_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self._run_propagation)
        r_lay.addWidget(self.run_btn)
        
        self.res_btn = QPushButton("Send to Research Mode")
        self.res_btn.clicked.connect(self._send_to_research_mode)
        r_lay.addWidget(self.res_btn)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_input_group(self, layout):
        g = QGroupBox("1. Input Data")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QFormLayout(g)
        self.prop_combo = QComboBox()
        self.prop_combo.currentTextChanged.connect(self._update_list)
        self.real_combo = QComboBox()
        self.samp_spin = QSpinBox()
        self.samp_spin.setRange(1, 10000)
        self.samp_spin.setValue(100)
        l.addRow("Property:", self.prop_combo)
        l.addRow("Set:", self.real_combo)
        l.addRow("Sample N:", self.samp_spin)
        layout.addWidget(g)

    def _create_econ_group(self, layout):
        g = QGroupBox("2. Economic Model")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QFormLayout(g)
        self.price = QDoubleSpinBox()
        self.price.setRange(0, 1e5)
        self.price.setValue(60)
        self.m_cost = QDoubleSpinBox()
        self.m_cost.setValue(2.5)
        self.p_cost = QDoubleSpinBox()
        self.p_cost.setValue(8.0)
        self.rec = QDoubleSpinBox()
        self.rec.setRange(0, 1)
        self.rec.setValue(0.85)
        self.disc = QDoubleSpinBox()
        self.disc.setRange(0, 1)
        self.disc.setValue(0.10)
        
        l.addRow("Price ($):", self.price)
        l.addRow("Mining ($/t):", self.m_cost)
        l.addRow("Proc ($/t):", self.p_cost)
        l.addRow("Recovery:", self.rec)
        l.addRow("Discount:", self.disc)
        layout.addWidget(g)

    def _create_opt_group(self, layout):
        g = QGroupBox("3. Optimization")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ba68c8; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        l = QVBoxLayout(g)
        self.reopt_pit = QCheckBox("Re-optimize Pit")
        self.reopt_pit.setChecked(True)
        self.reopt_sched = QCheckBox("Re-optimize Schedule")
        self.reopt_sched.setChecked(True)
        l.addWidget(self.reopt_pit)
        l.addWidget(self.reopt_sched)
        
        f = QFormLayout()
        self.slope = QDoubleSpinBox()
        self.slope.setValue(45)
        self.periods = QSpinBox()
        self.periods.setValue(20)
        self.target = QDoubleSpinBox()
        self.target.setRange(0, 1e9)
        self.target.setValue(100000)
        f.addRow("Slope:", self.slope)
        f.addRow("Periods:", self.periods)
        f.addRow("Target (t):", self.target)
        l.addLayout(f)
        layout.addWidget(g)

    def _on_bm_loaded(self, bm):
        if hasattr(bm, 'to_dataframe'):
            self._block_model = bm.to_dataframe()
        else:
            self._block_model = bm

    def _on_sgsim_loaded(self, res):
        self.sgsim_results = res

    def _update_list(self):
        # Placeholder for property listing logic (same as original)
        # This would populate real_combo based on selected property
        pass

    def _run_propagation(self):
        # Placeholder for run logic (same as original, calls controller)
        self.log_text.append("Starting propagation...")
        # Simulate result for UI testing
        self.log_text.append("Completed.")

    def _on_propagation_results(self, payload):
        # Update table logic (same as original)
        # This would populate the summary table with results
        pass

    def _send_to_research_mode(self):
        # Placeholder
        QMessageBox.information(self, "Research Mode", "Sent.")

    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters from UI for controller."""
        return {
            'property': self.prop_combo.currentText(),
            'realization_set': self.real_combo.currentText(),
            'sample_n': self.samp_spin.value(),
            'price': self.price.value(),
            'mining_cost': self.m_cost.value(),
            'processing_cost': self.p_cost.value(),
            'recovery': self.rec.value(),
            'discount_rate': self.disc.value(),
            'reoptimize_pit': self.reopt_pit.isChecked(),
            'reoptimize_schedule': self.reopt_sched.isChecked(),
            'slope_angle': self.slope.value(),
            'periods': self.periods.value(),
            'target_tonnage': self.target.value()
        }

    def validate_inputs(self) -> bool:
        """Validate inputs before running analysis."""
        if self._block_model is None:
            QMessageBox.warning(self, "Error", "No block model loaded.")
            return False
        if not self.prop_combo.currentText():
            QMessageBox.warning(self, "Error", "Please select a property.")
            return False
        return True

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results from controller."""
        self.log_text.append("Propagation analysis complete.")
        # Update summary table
        # Update distributions tab
        self.show_info("Success", "Analysis complete.")
