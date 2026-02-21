"""
Bench Design Panel - UI for bench design rules (STEP 27).
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QGroupBox, QLabel,
    QPushButton, QMessageBox, QTextEdit, QDoubleSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QLineEdit, QFormLayout
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


class BenchDesignPanel(BaseAnalysisPanel):
    """Panel for bench design rules."""
    # PanelManager metadata
    PANEL_ID = "BenchDesignPanel"
    PANEL_NAME = "BenchDesign Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "bench_design"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="bench_design")
        self.setWindowTitle("Bench Design")
        # Sizing is handled by BaseAnalysisPanel._setup_panel_sizing()
        
        self.design_rules = []
        self._block_model = None  # Use _block_model instead of block_model property
        
        # Subscribe to block model from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_loaded)
            
            # Prefer classified block model when available.
            existing_block_model = self.registry.get_classified_block_model()
            if not _has_model_data(existing_block_model):
                existing_block_model = self.registry.get_block_model()
            if _has_model_data(existing_block_model):
                self._on_block_model_loaded(existing_block_model)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        # Crash prevention: Handle None case (e.g., when DataRegistry clears)
        if block_model is None:
            logger.debug("Bench Design Panel received None block_model (likely cleared)")
            self._block_model = None  # Use _block_model instead of block_model property
            return
        
        logger.info("Bench Design Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        title = QLabel("Bench Design Rules")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Design rule input
        group = QGroupBox("Design Rule Input")
        form = QFormLayout()
        
        self.domain_code_edit = QLineEdit()
        form.addRow("Domain Code:", self.domain_code_edit)
        
        self.rock_mass_combo = QComboBox()
        self.rock_mass_combo.addItems(["Excellent", "Good", "Fair", "Poor", "Very Poor"])
        form.addRow("Rock Mass Class:", self.rock_mass_combo)
        
        self.min_fos = QDoubleSpinBox()
        self.min_fos.setRange(1.0, 3.0)
        self.min_fos.setValue(1.3)
        form.addRow("Min FOS Target:", self.min_fos)
        
        group.setLayout(form)
        layout.addWidget(group)
        
        btn = QPushButton("Suggest Design")
        btn.clicked.connect(self._suggest_design)
        layout.addWidget(btn)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Domain", "Rock Mass", "Bench Height", "Berm Width", "Slope Angle"])
        layout.addWidget(self.results_table)
        
        # Status
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        layout.addWidget(self.status_text)
    
    def _suggest_design(self):
        """Suggest bench design."""
        if not self.controller:
            QMessageBox.warning(self, "Error", "Controller not available")
            return
        
        # For now, create a simple material
        from ..geotech_common.material_properties import GeotechMaterial
        material = GeotechMaterial(
            name="default",
            unit_weight=25.0,
            friction_angle=35.0,
            cohesion=100.0
        )
        
        config = {
            "domain_code": self.domain_code_edit.text(),
            "material": material,
            "rock_mass_class": self.rock_mass_combo.currentText(),
            "constraints": {"min_fos_target": self.min_fos.value()}
        }
        
        self.controller.suggest_bench_design(config, self._on_design_complete)
        self.status_text.append("Suggesting bench design...")
    
    def _on_design_complete(self, result: Dict[str, Any]):
        """Handle design suggestion completion."""
        rule = result.get("rule")
        if rule:
            self.design_rules.append(rule)
            self._update_table()
            self.status_text.append(f"Design suggested: Bench height={rule.bench_height:.1f}m, Slope angle={rule.overall_slope_angle:.1f}°")
    
    def _update_table(self):
        """Update results table."""
        self.results_table.setRowCount(len(self.design_rules))
        for i, rule in enumerate(self.design_rules):
            self.results_table.setItem(i, 0, QTableWidgetItem(rule.domain_code))
            self.results_table.setItem(i, 1, QTableWidgetItem(rule.rock_mass_class))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{rule.bench_height:.1f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{rule.berm_width:.1f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{rule.overall_slope_angle:.1f}°"))

