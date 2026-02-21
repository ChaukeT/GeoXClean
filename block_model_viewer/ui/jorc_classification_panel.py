"""
JORC/SAMREC Resource Classification Panel (Redesigned)

======================================================

Professional-grade UI for resource classification.

Changes in this version:
- Implemented QSplitter for resizable workspaces.
- Added QScrollArea to Configuration pane to prevent "squeezing".
- Enforced Minimum Widths on cards to ensure sliders remain usable.
- Improved "ModernSlider" widget for better precision.
- Classification runs in background thread for responsiveness.
- Fixed Visualize 3D and Export Audit Report functionality.
"""

from __future__ import annotations

import logging
import csv
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .panel_manager import PanelCategory, DockArea
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox, 
    QSpinBox, QComboBox, QPushButton, QLabel, QMessageBox, 
    QWidget, QFrame, QProgressBar, QTableWidget, QTableWidgetItem, 
    QHeaderView, QSlider, QSplitter, QScrollArea, QSizePolicy, 
    QApplication, QToolButton, QCheckBox, QFileDialog, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from PyQt6.QtGui import QColor, QFont

from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from ..models.jorc_classification_engine import (
    JORCClassificationEngine,
    VariogramModel,
    ClassificationRuleset,
    ClassificationResult,
    CLASSIFICATION_COLORS,
    CLASSIFICATION_ORDER,
)
from ..utils.coordinate_utils import ensure_xyz_columns
from .modern_styles import ModernColors, get_theme_colors

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Background Worker for Classification (Non-blocking)
# ------------------------------------------------------------------ #

class ClassificationWorker(QObject):
    """Worker to run classification in background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # ClassificationResult
    error = pyqtSignal(str)
    
    def __init__(self, engine, block_data, drillhole_data):
        super().__init__()
        self.engine = engine
        self.block_data = block_data
        self.drillhole_data = drillhole_data

    def run(self):
        """Execute classification in background."""
        try:
            result = self.engine.classify(
                self.block_data, 
                self.drillhole_data,
                progress_callback=self._emit_progress
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Classification worker error")
            self.error.emit(str(e))
    
    def _emit_progress(self, pct: int, msg: str):
        """Emit progress signal (thread-safe)."""
        self.progress.emit(pct, msg)

class ModernSlider(QWidget):
    """
    A combined Slider + SpinBox for precise control.
    Layout: 
    [Label ........................ ] [Real Value Label]
    [Slider ----------------------- ] [SpinBox %]
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, min_pct: int, max_pct: int, default_pct: int, color: str, parent=None):
        super().__init__(parent)
        self.range_major = 100.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Row 1: Label and Real Value
        top_row = QHBoxLayout()
        self.lbl_title = QLabel(label)
        self.lbl_title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-weight: 500;")
        top_row.addWidget(self.lbl_title)
        
        top_row.addStretch()
        
        self.lbl_real_value = QLabel("= 0.0 m")
        self.lbl_real_value.setStyleSheet("color: #909090; font-family: monospace; font-weight: bold;")
        top_row.addWidget(self.lbl_real_value)
        layout.addLayout(top_row)
        
        # Row 2: Slider and Spinbox
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_pct, max_pct)
        self.slider.setValue(default_pct)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: #303038;
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
        """)
        
        self.spin = QSpinBox()
        self.spin.setRange(min_pct, max_pct)
        self.spin.setValue(default_pct)
        self.spin.setSuffix("%")
        self.spin.setFixedWidth(80)
        self.spin.setStyleSheet(f"""
            QSpinBox {{
                border: 1px solid #404040;
                background: #202020;
                color: {color};
                font-weight: bold;
                padding: 4px;
                border-radius: 4px;
            }}
        """)
        
        # Sync controls
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._on_change)
        
        ctrl_row.addWidget(self.slider)
        ctrl_row.addWidget(self.spin)
        layout.addLayout(ctrl_row)
        
        # Init label
        self._on_change(default_pct)

    def _on_change(self, val):
        real_dist = (val / 100.0) * self.range_major
        self.lbl_real_value.setText(f"= {real_dist:.1f} m")
        self.valueChanged.emit(val)
        
    def set_variogram_range(self, r):
        self.range_major = r
        self._on_change(self.slider.value())
        
    def value(self):
        return self.slider.value()

class ClassificationCategoryCard(QFrame):
    """
    Card widget for a single classification category.
    Enforces a minimum width to prevent squeezing.
    """
    parametersChanged = pyqtSignal()

    def __init__(self, category: str, color: str, default_dist_pct: int, default_min_holes: int, parent=None):
        super().__init__(parent)
        self.category = category
        self.color = color  # Store color for stylesheet
        self.setFrameShape(QFrame.Shape.StyledPanel)

        # Enforce minimum width so it doesn't get squeezed
        self.setMinimumWidth(400)

        self.setStyleSheet(self._get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        lbl_cat = QLabel(category.upper())
        lbl_cat.setStyleSheet(f"color: {color}; font-size: 11pt; font-weight: bold; letter-spacing: 1px;")
        header.addWidget(lbl_cat)
        layout.addLayout(header)
        
        # Slider Section
        self.dist_slider = ModernSlider(
            "Max Distance (% of Variogram Range)", 
            min_pct=5, max_pct=300, default_pct=default_dist_pct, color=color
        )
        self.dist_slider.valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.dist_slider)
        
        # Bottom Row: Holes and Options
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(20)
        
        # Holes
        holes_group = QHBoxLayout()
        lbl_holes = QLabel("Min Unique Holes:")
        lbl_holes.setStyleSheet("color: #ccc;")
        
        self.holes_spin = QSpinBox()
        self.holes_spin.setRange(1, 20)
        self.holes_spin.setValue(default_min_holes)
        self.holes_spin.setFixedWidth(60)
        self.holes_spin.setStyleSheet(f"background: #25252b; color: white; border: 1px solid {color}; padding: 4px;")
        self.holes_spin.valueChanged.connect(lambda: self.parametersChanged.emit())
        
        holes_group.addWidget(lbl_holes)
        holes_group.addWidget(self.holes_spin)
        bottom_row.addLayout(holes_group)
        
        # Optional KV Checkbox (only for Measured/Indicated)
        if category in ["Measured", "Indicated"]:
            self.chk_kv = QCheckBox("Limit KV")
            self.chk_kv.setToolTip("Enforce Kriging Variance limit (KV/Sill ratio)")
            self.chk_kv.setStyleSheet("color: #ccc;")
            self.chk_kv.toggled.connect(lambda: self.parametersChanged.emit())
            bottom_row.addWidget(self.chk_kv)
            
            # KV Spinbox (shown when checked)
            self.spin_kv = QSpinBox()
            self.spin_kv.setRange(10, 100)
            self.spin_kv.setValue(30 if category == "Measured" else 60)
            self.spin_kv.setSuffix("%")
            self.spin_kv.setFixedWidth(70)
            self.spin_kv.setToolTip("Maximum KV as % of sill")
            self.spin_kv.setStyleSheet(f"background: #25252b; color: #aaa; border: 1px solid #404040; padding: 2px;")
            self.spin_kv.valueChanged.connect(lambda: self.parametersChanged.emit())
            bottom_row.addWidget(self.spin_kv)
            
        bottom_row.addStretch()
        layout.addLayout(bottom_row)
        
        # SoR (Slope of Regression) row - only for Measured/Indicated
        if category in ["Measured", "Indicated"]:
            sor_row = QHBoxLayout()
            sor_row.setSpacing(10)
            
            self.chk_slope = QCheckBox("SoR Gate")
            self.chk_slope.setToolTip(
                "Enforce Slope-of-Regression minimum.\n"
                "Blocks below this slope are downgraded to lower category.\n"
                "Requires SLOPE column in block model (from kriging validation)."
            )
            self.chk_slope.setStyleSheet("color: #ccc;")
            self.chk_slope.toggled.connect(lambda: self.parametersChanged.emit())
            sor_row.addWidget(self.chk_slope)
            
            sor_row.addWidget(QLabel("Min SoR:"))
            self.spin_slope = QDoubleSpinBox()
            self.spin_slope.setRange(0.0, 1.20)
            self.spin_slope.setSingleStep(0.05)
            self.spin_slope.setDecimals(2)
            self.spin_slope.setValue(0.95 if category == "Measured" else 0.80)
            self.spin_slope.setFixedWidth(70)
            self.spin_slope.setToolTip("Minimum slope-of-regression (0.0-1.2)")
            self.spin_slope.setStyleSheet("background: #25252b; color: #aaa; border: 1px solid #404040; padding: 2px;")
            self.spin_slope.valueChanged.connect(lambda: self.parametersChanged.emit())
            sor_row.addWidget(self.spin_slope)
            
            sor_row.addStretch()
            layout.addLayout(sor_row)
        
        # Warning Label
        self.lbl_warning = QLabel()
        self.lbl_warning.setStyleSheet("color: #ff5252; font-style: italic; font-size: 9pt;")
        self.lbl_warning.setVisible(False)
        layout.addWidget(self.lbl_warning)

    def set_variogram_range(self, r):
        self.dist_slider.set_variogram_range(r)
        
    def set_warning(self, msg):
        self.lbl_warning.setText(msg)
        self.lbl_warning.setVisible(bool(msg))

    def get_parameters(self) -> Dict[str, Any]:
        """Get all classification parameters for this category."""
        params = {
            "dist_pct": self.dist_slider.value(),
            "min_holes": self.holes_spin.value(),
            # KV parameters
            "kv_enabled": self.chk_kv.isChecked() if hasattr(self, 'chk_kv') else False,
            "kv_pct": self.spin_kv.value() if hasattr(self, 'spin_kv') else (30 if self.category == "Measured" else 60),
            # SoR parameters
            "slope_enabled": self.chk_slope.isChecked() if hasattr(self, 'chk_slope') else False,
            "slope": self.spin_slope.value() if hasattr(self, 'spin_slope') else (0.95 if self.category == "Measured" else 0.80),
        }
        return params

    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
            ClassificationCategoryCard {{
                background-color: #1a1a20;
                border: 1px solid #303038;
                border-left: 6px solid {self.color};
                border-radius: 6px;
                margin-bottom: 10px;
            }}
        """

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            # Rebuild stylesheet with new theme colors
            self.setStyleSheet(self._get_stylesheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()

class JORCClassificationPanel(BaseAnalysisPanel):
    task_name = "jorc_classification"
    classification_complete = pyqtSignal(object)
    request_visualization = pyqtSignal(object, str)
    
    def __init__(self, parent=None):
        # Data State
        self.drillhole_data = None
        self.block_model_data = None
        self.classification_result = None
        self.cards = {}

        # Block model source selector storage
        self._block_model_sources: Dict[str, Any] = {}  # key -> {df, display_name, property, source_type}
        self._available_sources: list = []  # List of source keys in display order
        self._current_source: str = ""  # Currently selected source key

        super().__init__(parent=parent, panel_id="jorc_classification")
        self.setWindowTitle("Resource Classification")
        self._apply_theme()
        
        # Build UI
        self._build_ui()
        
        # Init Registry (connect to data signals and load existing data)
        self._init_registry()
    
    def _build_ui(self):
        """Build the UI layout."""
        # --- UI Construction ---
        # 1. Header
        self.main_layout.addWidget(self._create_header())
        
        # 2. Splitter (The "Two Window" feel)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.setStyleSheet("""
            QSplitter::handle { background-color: #303038; }
            QSplitter::handle:hover { background-color: #4da6ff; }
        """)
        
        # LEFT PANE: Configuration (Scrollable)
        self.left_pane = self._create_config_pane()
        self.splitter.addWidget(self.left_pane)
        
        # RIGHT PANE: Results (Static)
        self.right_pane = self._create_results_pane()
        self.splitter.addWidget(self.right_pane)
        
        # Set initial sizes (60% config, 40% results)
        self.splitter.setStretchFactor(0, 6)
        self.splitter.setStretchFactor(1, 4)
        
        self.main_layout.addWidget(self.splitter)
        
        # Initialize Values
        self._on_var_changed()
    
    def _setup_base_ui(self):
        """Override base class to skip scroll area wrapper - use direct layout."""
        # Create direct layout (no scroll area wrapper)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Don't call setup_ui() here - we'll build UI in __init__
        # But we still need to call connect_signals()
        self.connect_signals()
        
        self._is_initialized = True 

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QWidget {{ background-color: #0e1117; color: {ModernColors.TEXT_PRIMARY}; font-family: 'Segoe UI', sans-serif; }}
            QGroupBox {{ 
                border: 1px solid #303038; 
                margin-top: 20px; 
                border-radius: 4px;
                padding-top: 15px; 
                font-weight: bold;
            }}
            QGroupBox::title {{ 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
                color: #4da6ff;
            }}
            QScrollArea {{ border: none; background-color: transparent; }}
            QScrollBar:vertical {{
                border: none; background: #1a1a20; width: 12px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #404040; min-height: 20px; border-radius: 6px;
            }}
        """)

    def _create_header(self) -> QWidget:
        frame = QFrame()
        frame.setFixedHeight(60)
        frame.setStyleSheet("background-color: #151518; border-bottom: 2px solid #303038;")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        
        title = QLabel("Resource Classification Manager")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; color: white;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Refresh button to reload data from registry
        self.refresh_btn = QPushButton("⟳ Refresh")
        self.refresh_btn.setToolTip("Reload data from registry (Block Model, Drillholes, Variogram)")
        self.refresh_btn.setStyleSheet("""
            QPushButton { 
                background: #303038; color: #4da6ff; border: 1px solid #4da6ff; 
                padding: 5px 15px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background: #404050; }
        """)
        self.refresh_btn.clicked.connect(self._refresh_data)
        layout.addWidget(self.refresh_btn)
        
        layout.addSpacing(10)

        # Block Model Source Selector
        layout.addWidget(QLabel("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(280)
        self.source_combo.addItem("No block model loaded", "none")
        self.source_combo.setStyleSheet("background: #25252b; border: 1px solid #404040; padding: 5px;")
        self.source_combo.setToolTip("Select block model data source for classification")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        layout.addSpacing(10)

        layout.addWidget(QLabel("Domain:"))
        self.domain_combo = QComboBox()
        self.domain_combo.setMinimumWidth(200)
        self.domain_combo.addItem("Full Model Extent")
        self.domain_combo.setStyleSheet("background: #25252b; border: 1px solid #404040; padding: 5px;")
        layout.addWidget(self.domain_combo)

        layout.addSpacing(20)
        
        self.status_lbl = QLabel("WAITING FOR DATA")
        self.status_lbl.setStyleSheet("color: #777; font-weight: bold; background: #202020; padding: 5px 10px; border-radius: 4px;")
        layout.addWidget(self.status_lbl)
        
        return frame

    def _create_config_pane(self) -> QWidget:
        """Creates the scrollable left pane."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        
        # 1. Variogram Group
        var_group = QGroupBox("1. Variogram Parameters")
        var_layout = QGridLayout(var_group)
        var_layout.setVerticalSpacing(15)
        
        # Major Range
        self.spin_maj = QDoubleSpinBox()
        self.spin_maj.setRange(1, 99999)
        self.spin_maj.setValue(100)
        self.spin_maj.setSuffix(" m")
        self.spin_maj.setPrefix("Major: ")
        self.spin_maj.setStyleSheet("padding: 5px; background: #25252b;")
        self.spin_maj.valueChanged.connect(self._on_var_changed)
        var_layout.addWidget(self.spin_maj, 0, 0)
        
        # Semi Range
        self.spin_semi = QDoubleSpinBox()
        self.spin_semi.setRange(1, 99999)
        self.spin_semi.setValue(80)
        self.spin_semi.setSuffix(" m")
        self.spin_semi.setPrefix("Semi: ")
        self.spin_semi.setStyleSheet("padding: 5px; background: #25252b;")
        var_layout.addWidget(self.spin_semi, 0, 1)

        # Minor Range
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(1, 99999)
        self.spin_min.setValue(40)
        self.spin_min.setSuffix(" m")
        self.spin_min.setPrefix("Minor: ")
        self.spin_min.setStyleSheet("padding: 5px; background: #25252b;")
        var_layout.addWidget(self.spin_min, 1, 0)

        # Sill
        self.spin_sill = QDoubleSpinBox()
        self.spin_sill.setRange(0.01, 100)
        self.spin_sill.setValue(1.0)
        self.spin_sill.setPrefix("Sill: ")
        self.spin_sill.setStyleSheet("padding: 5px; background: #25252b;")
        var_layout.addWidget(self.spin_sill, 1, 1)
        
        # Load Button
        btn_load = QPushButton("Load from Variogram Analysis")
        btn_load.setStyleSheet("background: #303038; color: #4da6ff; border: 1px solid #4da6ff; padding: 6px; border-radius: 4px;")
        btn_load.clicked.connect(self._load_variogram_params)
        var_layout.addWidget(btn_load, 2, 0, 1, 2)
        
        layout.addWidget(var_group)
        
        # 2. Classification Cards
        lbl_cards = QLabel("2. Classification Thresholds")
        lbl_cards.setStyleSheet("color: #4da6ff; font-weight: bold; margin-top: 10px;")
        layout.addWidget(lbl_cards)
        
        # Measured
        self.cards["Measured"] = ClassificationCategoryCard(
            "Measured", CLASSIFICATION_COLORS["Measured"], 25, 3
        )
        self.cards["Measured"].parametersChanged.connect(self._validate_params)
        layout.addWidget(self.cards["Measured"])
        
        # Indicated
        self.cards["Indicated"] = ClassificationCategoryCard(
            "Indicated", CLASSIFICATION_COLORS["Indicated"], 60, 2
        )
        self.cards["Indicated"].parametersChanged.connect(self._validate_params)
        layout.addWidget(self.cards["Indicated"])
        
        # Inferred
        self.cards["Inferred"] = ClassificationCategoryCard(
            "Inferred", CLASSIFICATION_COLORS["Inferred"], 150, 1
        )
        self.cards["Inferred"].parametersChanged.connect(self._validate_params)
        layout.addWidget(self.cards["Inferred"])
        
        # Suggest Thresholds Button
        suggest_layout = QHBoxLayout()
        self.btn_suggest = QPushButton("✨ Suggest Thresholds")
        self.btn_suggest.setToolTip(
            "Analyze drillhole spacing and suggest optimal thresholds.\n"
            "Uses distance distributions to target coverage levels:\n"
            "• Measured: ~10% of blocks\n"
            "• Indicated: ~35% of blocks\n"
            "• Inferred: ~80% of blocks"
        )
        self.btn_suggest.setStyleSheet("""
            QPushButton {
                background: #2d5a27; color: #c5e1a5; 
                border: 1px solid #4caf50; padding: 8px 16px; 
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background: #3d7a37; }
            QPushButton:disabled { background: #25252b; color: #666; border: 1px solid #404040; }
        """)
        self.btn_suggest.clicked.connect(self._on_suggest_thresholds)
        self.btn_suggest.setEnabled(False)  # Enabled when data loaded
        suggest_layout.addWidget(self.btn_suggest)
        suggest_layout.addStretch()
        layout.addLayout(suggest_layout)
        
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _create_results_pane(self) -> QWidget:
        """Creates the right side results pane."""
        container = QWidget()
        container.setStyleSheet("background-color: #121215; border-left: 1px solid #303038;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        layout.addWidget(QLabel("3. Execution & Preview"))
        
        # Results Table
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["Category", "Blocks", "%"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget { background: #1a1a20; border: 1px solid #303038; }
            QHeaderView::section { background: #25252b; border: none; padding: 4px; }
        """)
        
        # Init Table
        for i, cat in enumerate(CLASSIFICATION_ORDER):
            item = QTableWidgetItem(cat)
            item.setForeground(QColor(CLASSIFICATION_COLORS[cat]))
            item.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            self.table.setItem(i, 0, item)
            self.table.setItem(i, 1, QTableWidgetItem("-"))
            self.table.setItem(i, 2, QTableWidgetItem("-"))
        
        # Alias for test compatibility
        self.results_table = self.table
            
        layout.addWidget(self.table)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { border: none; background: #25252b; height: 6px; } QProgressBar::chunk { background: #4da6ff; }")
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_lbl = QLabel("Ready")
        self.progress_lbl.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.progress_lbl)
        
        # Buttons
        layout.addStretch()
        
        self.btn_run = QPushButton("RUN CLASSIFICATION")
        self.btn_run.setFixedHeight(50)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32; color: white; border-radius: 4px; font-weight: bold; font-size: 11pt;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #25252b; color: #555; }
        """)
        self.btn_run.clicked.connect(self.run_classification)
        self.btn_run.setEnabled(False)
        layout.addWidget(self.btn_run)
        
        self.btn_viz = QPushButton("Visualize 3D")
        self.btn_viz.setFixedHeight(40)
        self.btn_viz.setStyleSheet("background-color: #1976d2; color: white; border-radius: 4px; font-weight: bold;")
        self.btn_viz.clicked.connect(self._visualize_results)
        self.btn_viz.setEnabled(False)
        layout.addWidget(self.btn_viz)
        
        btn_exp = QPushButton("Export Audit Report")
        btn_exp.setStyleSheet("background-color: #303038; color: #ddd; border: 1px solid #555; border-radius: 4px; padding: 8px;")
        btn_exp.clicked.connect(self._export_audit_report)
        layout.addWidget(btn_exp)

        return container

    # --- Logic ---

    def _init_registry(self):
        try:
            self.reg = self.get_registry()
            if self.reg:
                connected_signals = []
                
                # Helper to safely connect signals (handles None properties)
                def safe_connect(signal_name, handler):
                    signal = getattr(self.reg, signal_name, None)
                    if signal is not None:
                        try:
                            signal.connect(handler)
                            connected_signals.append(signal_name)
                            return True
                        except (TypeError, AttributeError) as e:
                            logger.debug(f"Could not connect {signal_name}: {e}")
                    return False
                
                # Drillhole data (includes composites from variogram/compositing)
                safe_connect('drillholeDataLoaded', self._on_dh_loaded)
                
                # Block model - loaded from file
                safe_connect('blockModelLoaded', self._on_bm_loaded)
                
                # Block model - generated by Block Model Builder
                safe_connect('blockModelGenerated', self._on_bm_loaded)
                
                # Block model - classified (can be used as input too)
                safe_connect('blockModelClassified', self._on_bm_loaded)
                
                # Variogram results (for loading variogram params)
                safe_connect('variogramResultsLoaded', self._on_var_loaded)
                
                # Estimation results - can be used as block model source
                estimation_signals = [
                    'krigingResultsLoaded',
                    'simpleKrigingResultsLoaded', 
                    'universalKrigingResultsLoaded',
                    'cokrigingResultsLoaded',
                    'indicatorKrigingResultsLoaded',
                    'softKrigingResultsLoaded',
                    'sgsimResultsLoaded',
                ]
                for sig_name in estimation_signals:
                    safe_connect(sig_name, self._on_estimation_results)
                
                # Schedule data load after UI is fully initialized
                QTimer.singleShot(500, self._load_existing)
                logger.info(f"JORC Classification: Connected {len(connected_signals)} signals: {connected_signals}")
        except Exception as e:
            logger.warning(f"JORC Classification: Failed to connect to registry: {e}", exc_info=True)

    def _on_estimation_results(self, results):
        """Handle estimation/simulation results as block model source.

        For SGSIM results, extracts individual statistics (MEAN, P10, P50, P90, STD)
        from the 'summary' dict as separate selectable block model sources.
        """
        logger.info(f"JORC Classification: Received estimation results")

        if not isinstance(results, dict):
            logger.warning(f"JORC Classification: Results is not a dict, type={type(results)}")
            return

        # Check if this is SGSIM results with summary statistics
        grid = results.get('grid') or results.get('pyvista_grid')
        params = results.get('params')
        summary = results.get('summary', {})
        variable = results.get('variable', results.get('property_name', 'estimate'))

        logger.info(f"JORC Classification: SGSIM results keys: {list(results.keys())}")
        logger.info(f"JORC Classification: Summary keys: {list(summary.keys()) if summary else 'None'}")
        logger.info(f"JORC Classification: params = {params is not None}")

        # Extract coordinates from grid or generate from params
        base_df = None
        n_blocks = 0

        if PYVISTA_AVAILABLE and grid is not None:
            try:
                import pyvista as pv
                if hasattr(grid, 'cell_centers'):
                    centers = grid.cell_centers()
                    points = centers.points
                    base_df = pd.DataFrame({
                        'X': points[:, 0],
                        'Y': points[:, 1],
                        'Z': points[:, 2],
                    })
                    n_blocks = len(base_df)
                    logger.info(f"JORC Classification: Extracted {n_blocks:,} cell centers from grid")

                    # Add block dimensions if available
                    if hasattr(grid, 'x') and hasattr(grid, 'y') and hasattr(grid, 'z'):
                        x_edges = np.asarray(grid.x)
                        y_edges = np.asarray(grid.y)
                        z_edges = np.asarray(grid.z)
                        if len(x_edges) > 1:
                            base_df['DX'] = np.median(np.diff(x_edges))
                        if len(y_edges) > 1:
                            base_df['DY'] = np.median(np.diff(y_edges))
                        if len(z_edges) > 1:
                            base_df['DZ'] = np.median(np.diff(z_edges))
            except Exception as e:
                logger.warning(f"JORC Classification: Failed to extract cell centers: {e}")

        # If no grid, generate coordinates from params
        if (base_df is None or base_df.empty) and params is not None:
            try:
                nx, ny, nz = params.nx, params.ny, params.nz
                xmin, ymin, zmin = params.xmin, params.ymin, params.zmin
                xinc, yinc, zinc = params.xinc, params.yinc, params.zinc

                # Generate cell center coordinates
                x_centers = np.arange(nx) * xinc + xmin + xinc / 2
                y_centers = np.arange(ny) * yinc + ymin + yinc / 2
                z_centers = np.arange(nz) * zinc + zmin + zinc / 2

                # Create meshgrid and flatten (Z varies fastest, then Y, then X)
                zz, yy, xx = np.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
                coords_x = xx.transpose(2, 1, 0).flatten()
                coords_y = yy.transpose(2, 1, 0).flatten()
                coords_z = zz.transpose(2, 1, 0).flatten()

                base_df = pd.DataFrame({'X': coords_x, 'Y': coords_y, 'Z': coords_z})
                base_df['DX'] = xinc
                base_df['DY'] = yinc
                base_df['DZ'] = zinc
                n_blocks = len(base_df)
                logger.info(f"JORC Classification: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
            except Exception as e:
                logger.warning(f"JORC Classification: Failed to generate coords from params: {e}")

        if base_df is None or base_df.empty:
            # Try fallback extraction
            df = self._extract_block_model_from_results(results)
            if df is not None and not df.empty:
                source_key = "sgsim_results"
                display_name = f"SGSIM Results - {len(df):,} blocks"
                self._register_source(source_key, df, display_name, variable, "sgsim", auto_select=False)
                self._update_source_selector()
                logger.info(f"JORC Classification: Registered SGSIM results from fallback extraction")
            else:
                logger.warning("JORC Classification: No grid, params, or extractable results found")
            return

        found_stats = []

        # Extract individual statistics from 'summary' dict
        # SGSIM stores: summary['mean'], summary['std'], summary['p10'], summary['p50'], summary['p90']
        stat_mapping = {
            'mean': ('sgsim_mean', 'SGSIM Mean', 'sgsim_mean'),
            'std': ('sgsim_std', 'SGSIM Std Dev', 'sgsim_std'),
            'p10': ('sgsim_p10', 'SGSIM P10', 'sgsim_p10'),
            'p50': ('sgsim_p50', 'SGSIM P50', 'sgsim_p50'),
            'p90': ('sgsim_p90', 'SGSIM P90', 'sgsim_p90'),
        }

        for stat_key, (key_prefix, display_prefix, source_type) in stat_mapping.items():
            stat_data = summary.get(stat_key)
            if stat_data is not None:
                stat_values = np.asarray(stat_data).flatten()
                if len(stat_values) == n_blocks:
                    df = base_df.copy()
                    prop_name = f"{variable}_{stat_key.upper()}"
                    df[prop_name] = stat_values

                    source_key = f"{key_prefix}_{variable}"
                    display_name = f"{display_prefix} ({variable}) - {n_blocks:,} blocks"
                    self._register_source(source_key, df, display_name, prop_name, source_type, auto_select=False)
                    found_stats.append(stat_key)
                    logger.info(f"JORC Classification: Registered {display_prefix} ({variable})")

        # Also extract from grid cell_data if available (e.g., FE_SGSIM_MEAN)
        if grid is not None and hasattr(grid, 'cell_data'):
            for prop_name in grid.cell_data.keys():
                prop_values = np.asarray(grid.cell_data[prop_name]).flatten()
                if len(prop_values) != n_blocks:
                    continue

                prop_upper = prop_name.upper()
                # Skip if we already have this statistic from summary
                if 'MEAN' in prop_upper and 'mean' in found_stats:
                    continue
                if 'STD' in prop_upper and 'std' in found_stats:
                    continue

                df = base_df.copy()
                df[prop_name] = prop_values

                # Categorize by statistic type
                if 'MEAN' in prop_upper or 'E_TYPE' in prop_upper:
                    source_key = f"sgsim_mean_{variable}"
                    display_name = f"SGSIM Mean ({variable}) - {n_blocks:,} blocks"
                    source_type = "sgsim_mean"
                else:
                    source_key = f"sgsim_{prop_name}"
                    display_name = f"SGSIM {prop_name} - {n_blocks:,} blocks"
                    source_type = "sgsim"

                if source_key not in self._block_model_sources:
                    self._register_source(source_key, df, display_name, prop_name, source_type, auto_select=False)
                    found_stats.append(prop_name)

        if found_stats:
            logger.info(f"JORC Classification: Registered {len(found_stats)} SGSIM statistics: {found_stats}")
            self._update_source_selector()

            # Auto-select MEAN if available
            mean_key = f"sgsim_mean_{variable}"
            if mean_key in self._block_model_sources and not self._current_source:
                self._select_source(mean_key)
        else:
            # Fallback: Extract as single block model source
            df = self._extract_block_model_from_results(results)
        if df is not None and not df.empty:
            source_key = "estimation_results"
            display_name = f"Estimation Results - {len(df):,} blocks"
            self._register_source(source_key, df, display_name, "estimate", "estimation", auto_select=True)
            self._update_source_selector()
            logger.info(f"JORC Classification: Loaded {len(df)} blocks from estimation results")
    
    def _extract_block_model_from_results(self, results) -> Optional[pd.DataFrame]:
        """Extract block model DataFrame from estimation/simulation results.
        
        Handles various result formats:
        - Kriging results: grid_x, grid_y, grid_z arrays + estimates
        - SGSIM results: grid arrays + realizations
        - DataFrame-based results: data, df, block_model keys
        """
        if results is None:
            return None
        
        if isinstance(results, dict):
            # Priority 1: Check for kriging/estimation grid arrays (grid_x, grid_y, grid_z, estimates)
            if all(k in results for k in ['grid_x', 'grid_y', 'grid_z']):
                try:
                    grid_x = np.asarray(results['grid_x']).flatten()
                    grid_y = np.asarray(results['grid_y']).flatten()
                    grid_z = np.asarray(results['grid_z']).flatten()
                    
                    # Get estimates or realizations
                    estimates = None
                    est_name = 'estimate'
                    if 'estimates' in results:
                        estimates = np.asarray(results['estimates']).flatten()
                        est_name = results.get('property_name', results.get('variable', 'estimate'))
                    elif 'realizations' in results:
                        # SGSIM - AUDIT WARNING: Multiple realizations require explicit handling
                        # Using E-type (mean) estimate - uncertainty information is lost!
                        reals = results['realizations']
                        if isinstance(reals, np.ndarray):
                            if reals.ndim > 1:
                                n_realizations = reals.shape[0]
                                logger.warning(
                                    f"JORC AUDIT WARNING: {n_realizations} simulation realizations found. "
                                    f"Computing E-type (mean) estimate for classification. "
                                    f"Uncertainty information is LOST. For full uncertainty analysis, "
                                    f"classify each realization separately."
                                )
                                estimates = np.mean(reals, axis=0).flatten()
                            else:
                                estimates = reals.flatten()
                        est_name = 'E_type_mean'  # Explicit naming to indicate averaged result
                    
                    # Build DataFrame
                    df_data = {'X': grid_x, 'Y': grid_y, 'Z': grid_z}
                    if estimates is not None and len(estimates) == len(grid_x):
                        df_data[est_name] = estimates
                    
                    # Add variances if available
                    if 'variances' in results:
                        variances = np.asarray(results['variances']).flatten()
                        if len(variances) == len(grid_x):
                            df_data['variance'] = variances
                    
                    df = pd.DataFrame(df_data)
                    if not df.empty:
                        logger.info(f"JORC Classification: Extracted {len(df):,} blocks from estimation grid arrays")
                        return ensure_xyz_columns(df)
                except Exception as e:
                    logger.warning(f"JORC Classification: Failed to extract grid arrays: {e}")
            
            # Priority 2: Check for DataFrame keys
            for key in ['data', 'df', 'block_model', 'results', 'grid', 'classified_df']:
                if key in results:
                    candidate = results[key]
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        logger.info(f"JORC Classification: Extracted {len(candidate):,} blocks from '{key}' key")
                        return ensure_xyz_columns(candidate)
            
            # Priority 3: Try the dict itself if it has X, Y, Z columns
            if all(k in results for k in ['X', 'Y', 'Z']):
                df = pd.DataFrame(results)
                if not df.empty:
                    return ensure_xyz_columns(df)
                    
        elif isinstance(results, pd.DataFrame):
            return ensure_xyz_columns(results)

        return None

    # --- Source Selector Helper Methods ---

    def _register_source(self, key: str, df: pd.DataFrame, display_name: str,
                         property_name: str = "", source_type: str = "unknown",
                         auto_select: bool = False):
        """Register a block model source for selection.

        Args:
            key: Unique identifier for this source
            df: DataFrame with block model data
            display_name: Human-readable name for the selector
            property_name: Name of the primary property/column
            source_type: Type of source (block_model, classified, sgsim_mean, etc.)
            auto_select: If True, automatically select this source
        """
        self._block_model_sources[key] = {
            'df': df,
            'display_name': display_name,
            'property': property_name,
            'source_type': source_type
        }

        if key not in self._available_sources:
            self._available_sources.append(key)

        logger.debug(f"JORC Classification: Registered source '{key}' ({source_type}) with {len(df):,} blocks")

        if auto_select:
            self._select_source(key)

    def _update_source_selector(self):
        """Update the source combo box with available sources."""
        if not hasattr(self, 'source_combo'):
            return

        self.source_combo.blockSignals(True)
        current_key = self._current_source

        self.source_combo.clear()

        if not self._available_sources:
            self.source_combo.addItem("No block model loaded", "none")
        else:
            # Add sources in priority order: Classified, Regular BM, Kriging, SGSIM stats
            priority_order = ['classified', 'block_model', 'kriging', 'estimation',
                              'sgsim_mean', 'sgsim_p10', 'sgsim_p50', 'sgsim_p90', 'sgsim_std', 'sgsim']

            sorted_sources = []
            for ptype in priority_order:
                for key in self._available_sources:
                    src = self._block_model_sources.get(key, {})
                    if src.get('source_type', '').startswith(ptype) and key not in sorted_sources:
                        sorted_sources.append(key)

            # Add any remaining sources not in priority order
            for key in self._available_sources:
                if key not in sorted_sources:
                    sorted_sources.append(key)

            # Add to combo box
            for key in sorted_sources:
                src = self._block_model_sources.get(key, {})
                display_name = src.get('display_name', key)
                self.source_combo.addItem(display_name, key)

        # Restore selection if possible
        if current_key:
            idx = self.source_combo.findData(current_key)
            if idx >= 0:
                self.source_combo.setCurrentIndex(idx)
            elif self.source_combo.count() > 0:
                self.source_combo.setCurrentIndex(0)
                self._current_source = self.source_combo.currentData()

        self.source_combo.blockSignals(False)

    def _select_source(self, key: str):
        """Programmatically select a source by key."""
        if key not in self._block_model_sources:
            return

        self._current_source = key
        src = self._block_model_sources[key]
        df = src.get('df')

        if df is not None and not df.empty:
            self.block_model_data = ensure_xyz_columns(df)
            logger.info(f"JORC Classification: Selected source '{key}' with {len(df):,} blocks")

        # Update combo box if it exists
        if hasattr(self, 'source_combo'):
            idx = self.source_combo.findData(key)
            if idx >= 0:
                self.source_combo.blockSignals(True)
                self.source_combo.setCurrentIndex(idx)
                self.source_combo.blockSignals(False)

        self._check_ready()

    def _on_source_changed(self, index: int):
        """Handle source selection change from combo box."""
        if not hasattr(self, 'source_combo'):
            return

        key = self.source_combo.currentData()
        if key == "none" or key is None:
            self.block_model_data = None
            self._current_source = ""
            self._check_ready()
            return

        if key in self._block_model_sources:
            self._select_source(key)
            src = self._block_model_sources[key]
            logger.info(f"JORC Classification: User selected '{src.get('display_name', key)}'")

            # Update domain combo for new block model
            self._update_domain_combo()

    def _update_domain_combo(self):
        """Update domain combo box with domains from current block model."""
        if not hasattr(self, 'domain_combo') or self.block_model_data is None:
            return

        self.domain_combo.clear()
        self.domain_combo.addItem("Full Model Extent")

        for col in ['DOMAIN', 'Domain', 'ZONE', 'Zone', 'domain', 'zone']:
            if col in self.block_model_data.columns:
                unique_domains = self.block_model_data[col].dropna().unique()
                for d in sorted(unique_domains):
                    self.domain_combo.addItem(str(d))
                logger.info(f"JORC Classification: Found {len(unique_domains)} domains in column '{col}'")
                break

    def _refresh_data(self):
        """Manual refresh button handler - reload all data from registry."""
        logger.info("JORC Classification: Manual refresh requested")
        self._load_existing()
        
        # Show feedback to user
        if self.drillhole_data is not None and self.block_model_data is not None:
            QMessageBox.information(self, "Refresh Complete", 
                f"Data refreshed successfully!\n\n"
                f"• Drillholes: {len(self.drillhole_data):,} samples\n"
                f"• Block Model: {len(self.block_model_data):,} blocks")
        elif self.drillhole_data is None and self.block_model_data is None:
            QMessageBox.warning(self, "No Data Found", 
                "No drillhole or block model data found in registry.\n\n"
                "Please ensure you have:\n"
                "1. Loaded drillhole data\n"
                "2. Built or loaded a block model")
        else:
            missing = []
            if self.drillhole_data is None:
                missing.append("• Drillhole data (run compositing or load assays)")
            if self.block_model_data is None:
                missing.append("• Block model (use Block Model Builder, run Kriging/SGSIM, or load from file)")
            QMessageBox.warning(self, "Missing Data", 
                f"Some data is still missing:\n\n" + "\n".join(missing))

    def _load_existing(self):
        """Load existing data from registry on panel open - registers ALL available sources."""
        if not hasattr(self, 'reg') or self.reg is None:
            logger.warning("JORC Classification: Registry not available")
            return

        logger.info("JORC Classification: Loading existing data from registry...")
        found_any = False

        # 1. Load drillhole data (prioritize composites)
        try:
            dh = self.reg.get_drillhole_data()
            if dh is not None:
                # Log what we found
                if isinstance(dh, dict):
                    comp = dh.get('composites')
                    assays = dh.get('assays')
                    comp_count = len(comp) if isinstance(comp, pd.DataFrame) and not comp.empty else 0
                    assay_count = len(assays) if isinstance(assays, pd.DataFrame) and not assays.empty else 0
                    logger.info(f"JORC Classification: Found drillhole data - {comp_count} composites, {assay_count} assays")
                self._on_dh_loaded(dh)
            else:
                logger.info("JORC Classification: No drillhole data in registry")
        except Exception as e:
            logger.warning(f"JORC Classification: Failed to load drillhole data: {e}")

        # 2. Load variogram results (for variogram parameters)
        try:
            if hasattr(self.reg, 'get_variogram_results'):
                var_results = self.reg.get_variogram_results()
                if var_results is not None:
                    self._on_var_loaded(var_results)
                    logger.info("JORC Classification: Found variogram results in registry")
        except Exception as e:
            logger.debug(f"JORC Classification: Failed to load variogram results: {e}")

        # 3. Try to get classified block model (highest priority)
        try:
            if hasattr(self.reg, 'get_classified_block_model'):
                bm = self.reg.get_classified_block_model()
                if bm is not None:
                    df = bm if isinstance(bm, pd.DataFrame) else (bm.to_dataframe() if hasattr(bm, 'to_dataframe') else None)
                    if df is not None and not df.empty:
                        df = ensure_xyz_columns(df)
                        n_blocks = len(df)
                        self._register_source("classified_block_model", df,
                                              f"Classified Block Model - {n_blocks:,} blocks",
                                              "", "classified", auto_select=True)
                        found_any = True
                        logger.info(f"JORC Classification: Found classified block model ({n_blocks:,} blocks)")
        except Exception as e:
            logger.debug(f"JORC Classification: Failed to load classified block model: {e}")

        # 4. Regular block model (from file or builder)
        try:
            bm = self.reg.get_block_model()
            if bm is not None:
                df = bm if isinstance(bm, pd.DataFrame) else (bm.to_dataframe() if hasattr(bm, 'to_dataframe') else None)
                if df is not None and not df.empty:
                    df = ensure_xyz_columns(df)
                    n_blocks = len(df)
                    self._register_source("block_model", df,
                                          f"Block Model - {n_blocks:,} blocks",
                                          "", "block_model", auto_select=not found_any)
                    found_any = True
                    logger.info(f"JORC Classification: Found block model ({n_blocks:,} blocks)")
        except Exception as e:
            logger.warning(f"JORC Classification: Failed to load block model: {e}")

        # 5. Try all estimation results (kriging, SGSIM, etc.)
        self._try_load_all_estimation_results(auto_select_first=not found_any)

        # 6. Update source selector UI
        self._update_source_selector()

        # 7. Log final status
        dh_status = "✓" if self.drillhole_data is not None else "✗"
        bm_status = "✓" if self.block_model_data is not None else "✗"
        n_sources = len(self._available_sources)
        logger.info(f"JORC Classification: Load complete - Drillholes: {dh_status}, Block Model: {bm_status}, Sources: {n_sources}")
    
    def _try_load_all_estimation_results(self, auto_select_first: bool = False):
        """Try to load ALL estimation/simulation results as separate sources.

        Args:
            auto_select_first: If True, auto-select the first successful source
        """
        if not hasattr(self, 'reg') or self.reg is None:
            return

        first_found = True

        # List of estimation result getters to try
        estimation_sources = [
            ('kriging_results', 'get_kriging_results', 'Ordinary Kriging', 'kriging'),
            ('simple_kriging_results', 'get_simple_kriging_results', 'Simple Kriging', 'kriging'),
            ('universal_kriging_results', 'get_universal_kriging_results', 'Universal Kriging', 'kriging'),
            ('cokriging_results', 'get_cokriging_results', 'Co-Kriging', 'kriging'),
            ('indicator_kriging_results', 'get_indicator_kriging_results', 'Indicator Kriging', 'kriging'),
            ('soft_kriging_results', 'get_soft_kriging_results', 'Soft Kriging', 'kriging'),
        ]

        for key, getter_name, source_name, source_type in estimation_sources:
            if hasattr(self.reg, getter_name):
                try:
                    results = getattr(self.reg, getter_name)()
                    if results is not None:
                        df = self._extract_block_model_from_results(results)
                        if df is not None and not df.empty:
                            n_blocks = len(df)
                            should_select = auto_select_first and first_found
                            self._register_source(key, df,
                                                  f"{source_name} - {n_blocks:,} blocks",
                                                  "", source_type, auto_select=should_select)
                            logger.info(f"JORC Classification: Registered {source_name} ({n_blocks:,} blocks)")
                            first_found = False
                except Exception as e:
                    logger.debug(f"JORC Classification: Failed to load {source_name}: {e}")

        # Handle SGSIM separately - use the signal handler to extract individual stats
        if hasattr(self.reg, 'get_sgsim_results'):
            try:
                results = self.reg.get_sgsim_results()
                if results is not None:
                    # This will register individual SGSIM statistics via _on_estimation_results
                    self._on_estimation_results(results)
            except Exception as e:
                logger.debug(f"JORC Classification: Failed to load SGSIM: {e}")

        if first_found:
            logger.info("JORC Classification: No estimation results found in registry")

    def _on_dh_loaded(self, data):
        """Load drillhole data, PRIORITIZING COMPOSITES over raw assays.
        
        Data sources:
        - Compositing Window: Saves composites to registry via drillholeDataLoaded signal
        - Variogram Panel: Uses composites from registry for analysis
        - Drillhole Import: Provides raw assay data
        """
        # Log diagnostic info about registry contents
        log_registry_data_status("JORC Classification", data)
        
        df = None
        source_type = "unknown"
        
        # Fix: Explicitly check for non-empty DataFrames to avoid ValueError
        if isinstance(data, dict):
            composites = data.get('composites')
            assays = data.get('assays')
            
            # Priority 1: Composites (from Compositing Window or other sources)
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                df = composites
                source_type = "composites"
                logger.info(f"JORC Classification: Using COMPOSITES data ({len(df)} samples)")
            # Priority 2: Raw assays as fallback
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
                source_type = "assays"
                logger.info(f"JORC Classification: Using RAW ASSAYS data ({len(df)} samples) - composites not available")
            else:
                logger.warning("JORC Classification: Drillhole dict received but no valid composites or assays found")
        elif isinstance(data, pd.DataFrame) and not data.empty:
            df = data
            source_type = "dataframe"
            logger.info(f"JORC Classification: Using DataFrame ({len(df)} samples)")
        else:
            logger.warning(f"JORC Classification: Invalid drillhole data type: {type(data)}")
        
        # Ensure proper coordinate columns
        if df is not None:
            self.drillhole_data = ensure_xyz_columns(df)
            logger.info(f"JORC Classification: Loaded {len(self.drillhole_data)} drillhole samples ({source_type})")
            logger.debug(f"JORC Classification: Drillhole columns: {list(self.drillhole_data.columns)[:10]}")
        else:
            self.drillhole_data = None
        
        self._check_ready()

    def _on_bm_loaded(self, bm):
        """Handle block model data (from file, builder, or estimation results).

        Data sources:
        - Block Model Loading: CSV/Parquet block model files
        - Block Model Builder: Generated from drillhole data + grid definition
        - Kriging/SGSIM: Estimation results as block model

        Registers the block model as a selectable source.
        """
        try:
            if bm is None:
                logger.warning("JORC Classification: Received None block model")
                return

            source_type = type(bm).__name__

            # Convert to DataFrame if needed
            if hasattr(bm, 'to_dataframe'):
                df = bm.to_dataframe()
                source_type = f"BlockModel.to_dataframe ({type(bm).__name__})"
            elif isinstance(bm, pd.DataFrame):
                df = bm
                source_type = "DataFrame"
            elif isinstance(bm, dict):
                df = self._extract_block_model_from_results(bm)
                source_type = "dict (estimation results)"
            else:
                logger.warning(f"JORC Classification: Unknown block model type: {type(bm)}")
                return

            if df is None or df.empty:
                logger.warning("JORC Classification: Block model DataFrame is empty")
                return

            df = ensure_xyz_columns(df)
            n_blocks = len(df)

            # Determine source key and display name based on type
            if 'CLASS_FINAL' in df.columns or 'CLASS_AUTO' in df.columns:
                source_key = "classified_block_model"
                display_name = f"Classified Block Model - {n_blocks:,} blocks"
                src_type = "classified"
            else:
                source_key = "block_model"
                display_name = f"Block Model - {n_blocks:,} blocks"
                src_type = "block_model"

            # Register as source and auto-select if no source is currently selected
            auto_select = not self._current_source or self._current_source == "none"
            self._register_source(source_key, df, display_name, "", src_type, auto_select=auto_select)

            # Update source selector UI
            self._update_source_selector()

            logger.info(f"JORC Classification: Loaded block model ({source_type}) with {n_blocks:,} blocks")
            logger.debug(f"JORC Classification: Block model columns: {list(df.columns)[:15]}")

            # Update domain combo
            self._update_domain_combo()

        except Exception as e:
            logger.error(f"JORC Classification: Error loading block model: {e}", exc_info=True)

    def _on_var_loaded(self, results):
        """Handle variogram results from registry."""
        if results is None:
            logger.warning("JORC Classification: Received None variogram results")
            return
        
        self.variogram_results = results
        logger.info(f"JORC Classification: Loaded variogram results (keys: {list(results.keys()) if isinstance(results, dict) else 'N/A'})")
        
        # Auto load if defaults (range == 100 means user hasn't changed it)
        if self.spin_maj.value() == 100: 
            self._load_variogram_params()

    def _check_ready(self):
        """Check if all required data is available and update UI status."""
        has_drillholes = self.drillhole_data is not None
        has_block_model = self.block_model_data is not None
        ready = has_drillholes and has_block_model
        
        self.btn_run.setEnabled(ready)
        self.btn_suggest.setEnabled(ready)  # Enable suggest when data available
        
        if ready:
            dh_count = len(self.drillhole_data)
            bm_count = len(self.block_model_data)
            self.status_lbl.setText(f"READY ({dh_count:,} samples, {bm_count:,} blocks)")
            self.status_lbl.setStyleSheet("color: #4caf50; font-weight: bold; background: #202020; padding: 5px 10px; border-radius: 4px;")
            logger.info(f"JORC Classification: READY - {dh_count} drillhole samples, {bm_count} blocks")
        else:
            # Show specific missing data
            missing = []
            if not has_drillholes:
                missing.append("DRILLHOLES")
            if not has_block_model:
                missing.append("BLOCK MODEL")
            
            status_text = f"MISSING: {', '.join(missing)}"
            self.status_lbl.setText(status_text)
            self.status_lbl.setStyleSheet("color: #ff5252; font-weight: bold; background: #202020; padding: 5px 10px; border-radius: 4px;")
            logger.info(f"JORC Classification: Not ready - {status_text}")

    def _on_var_changed(self):
        r = self.spin_maj.value()
        for card in self.cards.values():
            card.set_variogram_range(r)
        self._validate_params()

    def _load_variogram_params(self):
        if not hasattr(self, 'variogram_results') or not self.variogram_results:
            # Try to fetch from registry as fallback
            if hasattr(self, 'reg') and self.reg:
                try:
                    vario = self.reg.get_variogram_results()
                    if vario:
                        self.variogram_results = vario
                        logger.info("Fetched variogram results from registry")
                    else:
                        QMessageBox.information(
                            self, "Info",
                            "No variogram results available.\n\n"
                            "Please run variogram analysis first in the Variogram Panel."
                        )
                        return
                except Exception as e:
                    logger.warning(f"Failed to fetch variogram from registry: {e}")
                    QMessageBox.information(self, "Info", "No variogram results available.")
                    return
            else:
                QMessageBox.information(self, "Info", "No variogram results available.")
                return

        try:
            # Debug: Log what we have
            logger.info(f"Variogram results keys: {list(self.variogram_results.keys()) if isinstance(self.variogram_results, dict) else 'not a dict'}")

            model = self.variogram_results.get('combined_3d_model', {})
            if not model:
                # Show available keys to help debug
                available_keys = list(self.variogram_results.keys()) if isinstance(self.variogram_results, dict) else []
                QMessageBox.warning(
                    self, "Warning",
                    f"No combined 3D model found in variogram results.\n\n"
                    f"Available keys: {available_keys}\n\n"
                    f"Please ensure you have run variogram analysis and fitted a model."
                )
                return

            # Debug: Log model contents
            logger.info(f"Combined 3D model keys: {list(model.keys())}")

            # Extract the three directional ranges
            major_range = model.get('major_range', 100.0)
            minor_range = model.get('minor_range', 80.0)
            vertical_range = model.get('vertical_range', 40.0)

            # CRITICAL FIX: Sort ranges by magnitude (Major >= Semi >= Minor)
            # The variogram panel stores by DIRECTION (horizontal major, horizontal minor, vertical)
            # But classification needs them by MAGNITUDE for proper ellipsoid shape
            ranges = sorted([major_range, minor_range, vertical_range], reverse=True)
            sorted_major = ranges[0]  # Largest
            sorted_semi = ranges[1]   # Middle
            sorted_minor = ranges[2]  # Smallest

            # Log if reordering was needed
            if (sorted_major != major_range or sorted_semi != minor_range or sorted_minor != vertical_range):
                logger.info(
                    f"Range reordering applied for classification ellipsoid: "
                    f"({major_range:.1f}, {minor_range:.1f}, {vertical_range:.1f}) -> "
                    f"({sorted_major:.1f}, {sorted_semi:.1f}, {sorted_minor:.1f})"
                )

            self.spin_maj.setValue(sorted_major)
            self.spin_semi.setValue(sorted_semi)
            self.spin_min.setValue(sorted_minor)

            # CRITICAL FIX: Use total_sill, not partial sill
            # combined_3d_model['sill'] is PARTIAL sill (C)
            # combined_3d_model['total_sill'] is TOTAL sill (C0 + C)
            # VariogramModel expects total sill for KV ratio calculations
            total_sill = model.get('total_sill', model.get('sill', 1.0))
            self.spin_sill.setValue(total_sill)

            logger.info(
                f"Loaded variogram params: Major={sorted_major:.2f}m, Semi={sorted_semi:.2f}m, "
                f"Minor={sorted_minor:.2f}m, Total Sill={total_sill:.4f}"
            )

            # Update the UI to reflect new variogram range
            self._on_var_changed()

            # Show success message
            QMessageBox.information(
                self, "Variogram Loaded",
                f"Variogram parameters loaded successfully:\n\n"
                f"• Major Range: {sorted_major:.2f} m\n"
                f"• Semi Range: {sorted_semi:.2f} m\n"
                f"• Minor Range: {sorted_minor:.2f} m\n"
                f"• Total Sill: {total_sill:.4f}"
            )

        except Exception as e:
            logger.exception(f"Could not set variogram model parameters: {e}")
            QMessageBox.warning(self, "Error", f"Failed to load variogram parameters:\n{e}")

    def _validate_params(self):
        m = self.cards["Measured"].get_parameters()['dist_pct']
        i = self.cards["Indicated"].get_parameters()['dist_pct']
        inf = self.cards["Inferred"].get_parameters()['dist_pct']

        self.cards["Measured"].set_warning("")
        self.cards["Indicated"].set_warning("")
        self.cards["Inferred"].set_warning("")

        if i <= m:
            self.cards["Indicated"].set_warning("Indicated distance must be > Measured")
        if inf <= i:
            self.cards["Inferred"].set_warning("Inferred distance must be > Indicated")

    def _align_coordinate_systems(self, block_data: pd.DataFrame, dh_data: pd.DataFrame):
        """
        Align coordinate systems between block model and drillhole data.

        This prevents coordinate mismatch errors when one dataset is in UTM coordinates
        (~500,000m) and the other is in local coordinates (~0m).

        Returns:
            tuple: (aligned_block_data, aligned_dh_data) - both in the same coordinate system
        """
        import numpy as np

        # Make copies to avoid modifying original data
        block_df = block_data.copy()
        dh_df = dh_data.copy()

        # Detect coordinate columns
        block_x = "XC" if "XC" in block_df.columns else "X"
        block_y = "YC" if "YC" in block_df.columns else "Y"
        block_z = "ZC" if "ZC" in block_df.columns else "Z"

        dh_x = "X"
        dh_y = "Y"
        dh_z = "Z"

        # Check if we have the required columns
        if not all(c in block_df.columns for c in [block_x, block_y, block_z]):
            logger.warning(f"Block model missing coordinate columns. Columns: {list(block_df.columns)}")
            return block_df, dh_df

        if not all(c in dh_df.columns for c in [dh_x, dh_y, dh_z]):
            logger.warning(f"Drillhole data missing coordinate columns. Columns: {list(dh_df.columns)}")
            return block_df, dh_df

        # Compute centroids
        block_centroid = np.array([
            block_df[block_x].mean(),
            block_df[block_y].mean(),
            block_df[block_z].mean()
        ])

        dh_centroid = np.array([
            dh_df[dh_x].mean(),
            dh_df[dh_y].mean(),
            dh_df[dh_z].mean()
        ])

        # Compute centroid separation
        centroid_offset = np.linalg.norm(block_centroid - dh_centroid)

        # If centroids are far apart (> 1km), apply coordinate shift
        # This indicates one dataset is in UTM and the other is in local coords
        if centroid_offset > 1000.0:
            logger.warning(
                f"Coordinate system mismatch detected! "
                f"Centroid offset: {centroid_offset:.1f}m. "
                f"Aligning block model to drillhole coordinate system."
            )
            logger.info(f"Block centroid: {block_centroid}")
            logger.info(f"Drillhole centroid: {dh_centroid}")

            # Shift block model to align with drillhole centroid
            shift = dh_centroid - block_centroid
            block_df[block_x] = block_df[block_x] + shift[0]
            block_df[block_y] = block_df[block_y] + shift[1]
            block_df[block_z] = block_df[block_z] + shift[2]

            logger.info(f"Applied coordinate shift: {shift}")
            logger.info(f"New block centroid: [{block_df[block_x].mean():.2f}, {block_df[block_y].mean():.2f}, {block_df[block_z].mean():.2f}]")
        else:
            logger.debug(f"Coordinate systems appear aligned (centroid offset: {centroid_offset:.1f}m)")

        return block_df, dh_df

    def _on_suggest_thresholds(self):
        """
        Analyze drillhole spacing and suggest optimal classification thresholds.
        
        Uses distance distributions to target coverage levels:
        - Measured: ~10% of blocks (3 unique holes)
        - Indicated: ~35% of blocks (2 unique holes)
        - Inferred: ~80% of blocks (1 unique hole)
        """
        if self.drillhole_data is None or self.block_model_data is None:
            QMessageBox.warning(self, "No Data", "Load drillhole and block model data first.")
            return

        # === Pre-flight validation: Check data quality BEFORE expensive computation ===

        # Check drillhole count
        n_holes = len(self.drillhole_data)
        if 'HOLEID' in self.drillhole_data.columns:
            n_unique_holes = len(self.drillhole_data['HOLEID'].unique())
        elif 'HOLE_ID' in self.drillhole_data.columns:
            n_unique_holes = len(self.drillhole_data['HOLE_ID'].unique())
        else:
            n_unique_holes = n_holes

        if n_unique_holes < 5:
            QMessageBox.warning(
                self,
                "Insufficient Drillholes",
                f"Auto-suggest requires at least 5 unique drillholes.\n\n"
                f"Found: {n_unique_holes} drillholes\n\n"
                f"Add more drillhole data or manually set thresholds."
            )
            return

        # Check coordinate columns exist in drillholes
        required_dh_cols = ['X', 'Y', 'Z']
        missing_dh = [c for c in required_dh_cols if c not in self.drillhole_data.columns]
        if missing_dh:
            QMessageBox.critical(
                self,
                "Missing Coordinates",
                f"Drillhole data missing coordinate columns: {missing_dh}\n\n"
                f"Available columns: {list(self.drillhole_data.columns)[:20]}\n\n"
                f"Ensure drillholes have X, Y, Z coordinates."
            )
            return

        # Check blocks have coordinates
        required_bm_cols = ['XC', 'YC', 'ZC']
        if not all(c in self.block_model_data.columns for c in required_bm_cols):
            # Try alternate column names
            alt_cols = [['X', 'Y', 'Z'], ['x', 'y', 'z'], ['XCENTER', 'YCENTER', 'ZCENTER']]
            found = False
            for alt in alt_cols:
                if all(c in self.block_model_data.columns for c in alt):
                    found = True
                    break

            if not found:
                QMessageBox.critical(
                    self,
                    "Missing Coordinates",
                    f"Block model missing coordinate columns.\n\n"
                    f"Expected: XC/YC/ZC or X/Y/Z\n"
                    f"Available: {list(self.block_model_data.columns)[:20]}\n\n"
                    f"Ensure blocks have centroid coordinates."
                )
                return

        # Check for NaN coordinates
        dh_coords_valid = self.drillhole_data[['X', 'Y', 'Z']].notna().all(axis=1).sum()
        if dh_coords_valid < n_unique_holes * 0.5:
            QMessageBox.warning(
                self,
                "Invalid Coordinates",
                f"More than 50% of drillhole samples have NaN coordinates!\n\n"
                f"Valid samples: {dh_coords_valid} / {len(self.drillhole_data)}\n\n"
                f"Clean your drillhole data before running auto-suggest."
            )
            return

        # Check variogram range is reasonable
        var_range = self.spin_maj.value()
        if var_range < 1.0:
            QMessageBox.warning(
                self,
                "Invalid Variogram",
                f"Variogram major range is too small: {var_range}m\n\n"
                f"Set a realistic variogram range before auto-suggesting thresholds."
            )
            return

        # Warn if range seems unrealistically large
        if var_range > 10000:
            reply = QMessageBox.question(
                self,
                "Large Variogram Range",
                f"Variogram major range is very large: {var_range:,.0f}m\n\n"
                f"This seems unusually large. Typical ranges are 50-500m.\n\n"
                f"Continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        logger.info(f"Pre-flight validation passed: {n_unique_holes} drillholes, "
                   f"{len(self.block_model_data)} blocks, variogram range={var_range}m")

        # === End pre-flight validation ===

        self.btn_suggest.setEnabled(False)
        n_blocks = len(self.block_model_data) if self.block_model_data is not None else 0
        sample_note = " (using fast sampling)" if n_blocks > 10_000 else ""
        self.progress_lbl.setText(f"Analyzing drillhole spacing{sample_note}...")
        self.progress_bar.setValue(0)
        # Force UI update using safer processEvents
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        
        try:
            from ..models.jorc_classification_engine import (
                JORCClassificationEngine, VariogramModel, ClassificationRuleset,
                suggest_thresholds_from_distances
            )
            
            # Build variogram model from current UI values
            var = VariogramModel(
                range_major=self.spin_maj.value(),
                range_semi=self.spin_semi.value(),
                range_minor=self.spin_min.value(),
                sill=self.spin_sill.value()
            )
            
            # Create engine with dummy loose rules (values don't matter for diagnostics)
            dummy_rules = ClassificationRuleset.from_ui_params(
                meas_dist_pct=300, meas_min_holes=3,
                ind_dist_pct=300, ind_min_holes=2,
                inf_dist_pct=300, inf_min_holes=1,
            )
            
            engine = JORCClassificationEngine(variogram=var, ruleset=dummy_rules)
            
            # Progress callback (throttled - only update every 10%)
            _last_pct = [0]
            def progress_cb(pct, msg):
                if pct >= _last_pct[0] + 10 or pct >= 100:
                    self.progress_bar.setValue(pct)
                    self.progress_lbl.setText(msg)
                    _last_pct[0] = pct
                    # Force UI repaint using safer processEvents
                    from PyQt6.QtCore import QEventLoop
                    from PyQt6.QtWidgets import QApplication
                    QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

            # === COORDINATE ALIGNMENT FIX ===
            # Align coordinate systems before computing distances
            # This prevents coordinate mismatch errors when one dataset is in UTM and the other is in local coords
            block_data_aligned, dh_data_aligned = self._align_coordinate_systems(
                self.block_model_data,
                self.drillhole_data
            )

            # Compute distance diagnostics
            dist_1st, dist_2nd, dist_3rd = engine.compute_distance_diagnostics(
                block_data_aligned,
                dh_data_aligned,
                progress_callback=progress_cb
            )
            
            # Get suggested thresholds
            suggestions = suggest_thresholds_from_distances(dist_1st, dist_2nd, dist_3rd)

            # === Validate suggestions before applying ===
            if suggestions is None:
                raise ValueError("suggest_thresholds_from_distances returned None")

            # Check for error diagnostics
            diag = suggestions.get("diagnostics", {})
            if "error" in diag:
                error_msg = diag["error"]
                hint = diag.get("hint", "")
                median_dist = diag.get("median_distance")

                if median_dist and median_dist > 100.0:
                    # Coordinate mismatch detected
                    QMessageBox.critical(
                        self,
                        "Coordinate System Mismatch",
                        f"⚠️ CRITICAL: Blocks and drillholes appear to be in different coordinate systems!\n\n"
                        f"Median distance: {median_dist:.1f} × variogram range\n"
                        f"(This is impossibly large - indicates ~{median_dist * self.spin_maj.value():.0f}m separation)\n\n"
                        f"Possible causes:\n"
                        f"• Blocks in UTM coordinates (500,000m) but drillholes in Local (0m)\n"
                        f"• Blocks in Local but drillholes in UTM\n"
                        f"• Incorrect variogram range (too small)\n\n"
                        f"Fix:\n"
                        f"1. Check coordinate columns in both datasets\n"
                        f"2. Ensure both use same coordinate system\n"
                        f"3. Use coordinate transformation if needed\n\n"
                        f"Using default thresholds for now (25%, 60%, 100%)."
                    )
                else:
                    # Other error (e.g., empty arrays)
                    QMessageBox.warning(
                        self,
                        "Auto-Suggest Failed",
                        f"Could not compute reliable thresholds:\n\n{error_msg}\n\n{hint}\n\n"
                        f"Using default thresholds (25%, 60%, 100%)."
                    )

                self.progress_lbl.setText("Using default thresholds (error detected)")
                # Fall through to apply defaults (suggestions already has defaults)

            # === Validate suggestion values ===
            for cat_name in ["measured", "indicated", "inferred"]:
                if cat_name not in suggestions:
                    raise ValueError(f"Missing category '{cat_name}' in suggestions")
                if "dist_pct" not in suggestions[cat_name]:
                    raise ValueError(f"Missing 'dist_pct' for category '{cat_name}'")
                if "min_holes" not in suggestions[cat_name]:
                    raise ValueError(f"Missing 'min_holes' for category '{cat_name}'")

                # Validate range
                dist_pct = suggestions[cat_name]["dist_pct"]
                if not isinstance(dist_pct, (int, float)) or dist_pct < 0 or dist_pct > 500:
                    logger.error(f"Invalid dist_pct for {cat_name}: {dist_pct}")
                    suggestions[cat_name]["dist_pct"] = 25 if cat_name == "measured" else (60 if cat_name == "indicated" else 100)

            # Apply to UI - set all sliders with suggested values
            logger.info(f"Applying auto-suggestions: "
                       f"Measured={suggestions['measured']['dist_pct']}%, "
                       f"Indicated={suggestions['indicated']['dist_pct']}%, "
                       f"Inferred={suggestions['inferred']['dist_pct']}%")

            self.cards["Measured"].dist_slider.slider.setValue(suggestions["measured"]["dist_pct"])
            self.cards["Measured"].holes_spin.setValue(suggestions["measured"]["min_holes"])

            self.cards["Indicated"].dist_slider.slider.setValue(suggestions["indicated"]["dist_pct"])
            self.cards["Indicated"].holes_spin.setValue(suggestions["indicated"]["min_holes"])

            self.cards["Inferred"].dist_slider.slider.setValue(suggestions["inferred"]["dist_pct"])
            self.cards["Inferred"].holes_spin.setValue(suggestions["inferred"]["min_holes"])

            # Force UI repaint BEFORE showing message box so user sees updated sliders
            QApplication.processEvents()

            # Show diagnostics (only if no errors)
            if not diag.get("error"):
                n_blocks = len(self.block_model_data)
                sample_note = f"\n(Based on 10,000 block sample of {n_blocks:,} total)" if n_blocks > 10_000 else ""
                msg = (
                    f"Suggested thresholds applied!\n\n"
                    f"Distance medians (isotropic):\n"
                    f"  • 1st hole: {diag.get('dist_1st_median', 0):.3f}\n"
                    f"  • 2nd hole: {diag.get('dist_2nd_median', 0):.3f}\n"
                    f"  • 3rd hole: {diag.get('dist_3rd_median', 0):.3f}\n\n"
                    f"Expected coverage:\n"
                    f"  • Measured: {diag.get('actual_measured_coverage', 0):.1%}\n"
                    f"  • Indicated: {diag.get('actual_indicated_coverage', 0):.1%}\n"
                    f"  • Inferred: {diag.get('actual_inferred_coverage', 0):.1%}\n\n"
                    f"Review and adjust thresholds as needed for your deposit.{sample_note}"
                )
                self.progress_lbl.setText("Thresholds suggested - review and adjust as needed")
                QMessageBox.information(self, "Auto-Suggested Thresholds", msg)
            else:
                self.progress_lbl.setText("Defaults applied (data issue detected)")
            
            logger.info(f"Auto-suggested thresholds: {suggestions}")
            
        except Exception as e:
            logger.exception("Failed to suggest thresholds")
            QMessageBox.critical(self, "Error", f"Failed to suggest thresholds:\n{e}")
            self.progress_lbl.setText("Error")
        finally:
            self.btn_suggest.setEnabled(True)

    def run_classification(self):
        """Run classification in background thread to keep UI responsive."""
        self.btn_run.setEnabled(False)
        self.btn_viz.setEnabled(False)
        self.progress_lbl.setText("Initializing...")
        self.progress_bar.setValue(0)
        
        try:
            # 1. Build variogram model
            var = VariogramModel(
                range_major=self.spin_maj.value(),
                range_semi=self.spin_semi.value(),
                range_minor=self.spin_min.value(),
                sill=self.spin_sill.value()
            )
            
            p_meas = self.cards["Measured"].get_parameters()
            p_ind = self.cards["Indicated"].get_parameters()
            p_inf = self.cards["Inferred"].get_parameters()
            
            dom = self.domain_combo.currentText()
            dom_val = dom if dom != "Full Model Extent" else None
            
            # Build ruleset with all parameters (distance, holes, KV, SoR)
            rules = ClassificationRuleset.from_ui_params(
                # Distance thresholds
                meas_dist_pct=p_meas['dist_pct'], 
                meas_min_holes=p_meas['min_holes'],
                ind_dist_pct=p_ind['dist_pct'], 
                ind_min_holes=p_ind['min_holes'],
                inf_dist_pct=p_inf['dist_pct'], 
                inf_min_holes=p_inf['min_holes'],
                # KV gating
                meas_kv_enabled=p_meas.get('kv_enabled', False),
                meas_kv_pct=p_meas.get('kv_pct', 30),
                ind_kv_enabled=p_ind.get('kv_enabled', False),
                ind_kv_pct=p_ind.get('kv_pct', 60),
                # SoR gating
                meas_slope_enabled=p_meas.get('slope_enabled', False),
                meas_slope=p_meas.get('slope', 0.95),
                ind_slope_enabled=p_ind.get('slope_enabled', False),
                ind_slope=p_ind.get('slope', 0.80),
                # Domain
                domain_name=dom_val
            )
            
            # 2. Build engine
            engine = JORCClassificationEngine(
                variogram=var, ruleset=rules,
                domain_column="DOMAIN" if dom_val else None,
                domain_value=dom_val
            )
            
            # Store engine for export
            self._current_engine = engine

            # === COORDINATE ALIGNMENT FIX ===
            # Align coordinate systems before classification
            block_data_aligned, dh_data_aligned = self._align_coordinate_systems(
                self.block_model_data,
                self.drillhole_data
            )

            # 3. Create worker and thread
            self._worker_thread = QThread()
            self._worker = ClassificationWorker(
                engine,
                block_data_aligned.copy(),  # Use aligned data
                dh_data_aligned.copy()
            )
            self._worker.moveToThread(self._worker_thread)
            
            # 4. Connect signals
            self._worker_thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.error.connect(self._on_worker_error)
            self._worker.finished.connect(self._worker_thread.quit)
            self._worker.error.connect(self._worker_thread.quit)
            self._worker_thread.finished.connect(self._cleanup_worker)
            
            # 5. Start
            logger.info("Starting classification in background thread...")
            self._worker_thread.start()
            
        except Exception as e:
            logger.exception("Failed to start classification")
            QMessageBox.critical(self, "Error", str(e))
            self.progress_lbl.setText("Error")
            self.btn_run.setEnabled(True)
    
    def _on_worker_progress(self, pct: int, msg: str):
        """Handle progress updates from worker (thread-safe via signal)."""
        self.progress_bar.setValue(pct)
        self.progress_lbl.setText(msg)
    
    def _on_worker_finished(self, result):
        """Handle classification completion."""
        self.classification_result = result
        self._update_table()
        self.classification_complete.emit(result)
        self.btn_viz.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.progress_lbl.setText(f"Complete ({result.execution_time_seconds:.1f}s)")
        logger.info(f"Classification complete: {result.execution_time_seconds:.2f}s")
    
    def _on_worker_error(self, error_msg: str):
        """Handle classification error."""
        QMessageBox.critical(self, "Classification Error", error_msg)
        self.progress_lbl.setText("Error")
        self.btn_run.setEnabled(True)
    
    def _cleanup_worker(self):
        """Cleanup worker thread."""
        if hasattr(self, '_worker'):
            self._worker.deleteLater()
            self._worker = None
        if hasattr(self, '_worker_thread'):
            self._worker_thread.deleteLater()
            self._worker_thread = None

    def _update_table(self):
        """Update the results table with classification summary."""
        if not self.classification_result:
            return
        
        try:
            summary = self.classification_result.summary
            if not summary:
                logger.warning("Classification result summary is empty")
                return
            
            for i, cat in enumerate(CLASSIFICATION_ORDER):
                # Get category data from summary
                d = summary.get(cat, {'count': 0, 'percentage': 0})
                
                # Ensure we have count and percentage
                count = d.get('count', 0) if isinstance(d, dict) else 0
                percentage = d.get('percentage', 0.0) if isinstance(d, dict) else 0.0
                
                # Update table items (ensure they exist)
                if self.table.item(i, 1) is None:
                    self.table.setItem(i, 1, QTableWidgetItem(""))
                if self.table.item(i, 2) is None:
                    self.table.setItem(i, 2, QTableWidgetItem(""))
                
                self.table.item(i, 1).setText(f"{count:,}")
                self.table.item(i, 2).setText(f"{percentage:.1f}%")
                
            logger.info(f"Updated results table with classification summary")
        except Exception as e:
            logger.error(f"Error updating results table: {e}", exc_info=True)
            QMessageBox.warning(self, "Update Error", f"Could not update results table: {e}")

    def _find_sgsim_grid_in_viewer(self):
        """Locate the SGSIM PyVista grid currently displayed in the 3D viewer.

        Returns
        -------
        pyvista grid or None
        """
        try:
            from PyQt6.QtWidgets import QApplication
            main_window = QApplication.instance().activeWindow()
            if main_window is None or not hasattr(main_window, 'viewer_widget'):
                return None

            renderer = getattr(main_window.viewer_widget, 'renderer', None)
            if renderer is None or not hasattr(renderer, 'active_layers'):
                return None

            for layer_name, layer_data in renderer.active_layers.items():
                if 'SGSIM' in layer_name and 'data' in layer_data:
                    grid = layer_data['data']
                    logger.info(f"Found SGSIM grid in viewer: {layer_name} "
                                f"({type(grid).__name__}, {grid.n_cells} cells)")
                    return grid
        except Exception as e:
            logger.debug(f"Could not locate SGSIM grid in viewer: {e}")
        return None

    def _visualize_results(self):
        """Visualize classification results in 3D viewer.

        Strategy
        --------
        1. **Clone SGSIM grid** — if the SGSIM grid is in the viewer we copy
           its exact geometry (edges / points) so the classification grid is
           pixel-perfect aligned.  Classification values are mapped onto grid
           cells by flat index (both come from the same SGSIM run, same
           nx×ny×nz ordering).
        2. **Fallback** — if the SGSIM grid is not available, build a new
           RectilinearGrid (regular) or UnstructuredGrid (irregular) from the
           classified DataFrame coordinates.
        """
        if not self.classification_result:
            QMessageBox.warning(self, "No Results", "Run classification first.")
            return

        if not PYVISTA_AVAILABLE:
            QMessageBox.warning(self, "PyVista Required",
                "PyVista is required for 3D visualization.\nInstall with: pip install pyvista")
            return

        try:
            df = self.classification_result.classified_df.copy()

            if df.empty:
                QMessageBox.warning(self, "No Data", "No classified blocks to visualize.")
                return

            # ── Classification value preparation ──────────────────────────
            class_col = "CLASS_FINAL" if "CLASS_FINAL" in df.columns else "CLASS_AUTO"
            if class_col not in df.columns:
                logger.error(f"Classification column not found. Available: {list(df.columns)}")
                QMessageBox.warning(self, "Column Error",
                    f"Classification column '{class_col}' not found.")
                return

            cat_map = {"Measured": 0, "Indicated": 1, "Inferred": 2, "Unclassified": 3}
            class_values = df[class_col].map(cat_map).fillna(3).astype(np.int32).values
            cat_labels = df[class_col].values
            n_blocks = len(df)

            # Log distribution
            unique_cats, counts = np.unique(class_values, return_counts=True)
            cat_names = {0: "Measured", 1: "Indicated", 2: "Inferred", 3: "Unclassified"}
            dist_str = ", ".join([f"{cat_names.get(c, c)}: {n}" for c, n in zip(unique_cats, counts)])
            logger.info(f"Classification distribution ({n_blocks:,} blocks): {dist_str}")

            # ==============================================================
            # STRATEGY 1: Clone the SGSIM grid from the viewer
            # ==============================================================
            # This guarantees perfect alignment because we reuse the exact
            # same grid geometry that is already displayed.
            # ==============================================================
            sgsim_grid = self._find_sgsim_grid_in_viewer()
            grid = None

            if sgsim_grid is not None and sgsim_grid.n_cells == n_blocks:
                logger.info(
                    f"✅ Cloning SGSIM grid geometry ({sgsim_grid.n_cells} cells) "
                    f"for classification overlay — guaranteed alignment"
                )
                grid = sgsim_grid.copy(deep=True)

                # Strip old cell data and populate with classification
                for name in list(grid.cell_data.keys()):
                    del grid.cell_data[name]

                grid.cell_data["Classification"] = class_values
                grid.cell_data["CLASS"] = class_values
                grid.cell_data["Category"] = cat_labels

                # Copy diagnostic columns
                for col in ["DIST_REAL_1ST", "N_HOLES_MEAS", "N_HOLES_IND", "N_HOLES_INF"]:
                    if col in df.columns:
                        grid.cell_data[col] = df[col].values

                # Preserve the shifted-coordinate flag from the source grid
                if getattr(sgsim_grid, '_coordinate_shifted', False):
                    grid._coordinate_shifted = True
                    logger.info("✅ Inherited _coordinate_shifted flag from SGSIM grid")

            elif sgsim_grid is not None and sgsim_grid.n_cells != n_blocks:
                logger.warning(
                    f"SGSIM grid has {sgsim_grid.n_cells} cells but classified_df "
                    f"has {n_blocks} rows — falling back to coordinate-based grid. "
                    f"(domain filtering may have removed blocks)"
                )

            # ==============================================================
            # STRATEGY 2: Fallback – build grid from DataFrame coordinates
            # ==============================================================
            if grid is None:
                grid = self._build_classification_grid_from_df(
                    df, class_values, cat_labels, sgsim_grid
                )

            if grid is None:
                QMessageBox.warning(self, "Grid Error",
                    "Failed to build classification grid.")
                return

            # Set active scalars for coloring
            grid.set_active_scalars("Classification")

            # Emit the grid for rendering
            logger.info(f"Emitting classification grid: {grid.n_cells} cells, {grid.n_points} points")
            self.request_visualization.emit(grid, "Resource Classification")
            self.progress_lbl.setText(f"Visualized {n_blocks:,} classified blocks")

        except Exception as e:
            logger.exception("Visualization error")
            QMessageBox.critical(self, "Visualization Error", f"Failed to visualize:\n{e}")

    # ------------------------------------------------------------------
    def _build_classification_grid_from_df(
        self, df, class_values, cat_labels, sgsim_grid
    ):
        """Build a PyVista grid from the classified DataFrame (fallback path).

        When the SGSIM grid cell count differs from the classified_df (e.g.
        after domain filtering), we build a new grid.  If the SGSIM grid is
        available we use its *bounds* (min-corner approach, not centroid) to
        apply a precise coordinate shift so that the new grid aligns with the
        viewer.

        Returns
        -------
        pyvista grid or None
        """
        # ── Coordinate columns ────────────────────────────────────────
        x_col = next((c for c in ['XC', 'X', 'x', 'xc', 'XCENTER'] if c in df.columns), None)
        y_col = next((c for c in ['YC', 'Y', 'y', 'yc', 'YCENTER'] if c in df.columns), None)
        z_col = next((c for c in ['ZC', 'Z', 'z', 'zc', 'ZCENTER'] if c in df.columns), None)

        if not all([x_col, y_col, z_col]):
            logger.error(f"Missing XYZ columns. Available: {list(df.columns)[:15]}")
            return None

        # ── Coordinate shift: use min-corner alignment, not centroids ─
        # Centroid-based shifts break when the classification covers a
        # domain subset.  Min-corner alignment is robust because both
        # grids share the same block origin.
        coordinate_shift = None
        if sgsim_grid is not None:
            try:
                gb = sgsim_grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                # SGSIM grid bounds are *edge* bounds (cell face), so the
                # minimum cell center is at  edge_min + dx/2.  Similarly the
                # df coords are cell centers.  Shift = grid_min_center - df_min_center.
                df_xmin = df[x_col].min()
                df_ymin = df[y_col].min()
                df_zmin = df[z_col].min()

                # Estimate cell spacing in the viewer grid
                if hasattr(sgsim_grid, 'x') and len(np.asarray(sgsim_grid.x)) > 1:
                    sg_dx = float(np.median(np.diff(np.asarray(sgsim_grid.x))))
                    sg_dy = float(np.median(np.diff(np.asarray(sgsim_grid.y))))
                    sg_dz = float(np.median(np.diff(np.asarray(sgsim_grid.z))))
                else:
                    # Fallback: estimate from bounds & df unique counts
                    x_unique = np.sort(df[x_col].unique())
                    y_unique = np.sort(df[y_col].unique())
                    z_unique = np.sort(df[z_col].unique())
                    sg_dx = np.median(np.diff(x_unique)) if len(x_unique) > 1 else 10.0
                    sg_dy = np.median(np.diff(y_unique)) if len(y_unique) > 1 else 10.0
                    sg_dz = np.median(np.diff(z_unique)) if len(z_unique) > 1 else 5.0

                grid_min_center_x = gb[0] + sg_dx / 2
                grid_min_center_y = gb[2] + sg_dy / 2
                grid_min_center_z = gb[4] + sg_dz / 2

                coordinate_shift = np.array([
                    grid_min_center_x - df_xmin,
                    grid_min_center_y - df_ymin,
                    grid_min_center_z - df_zmin,
                ])

                # Only apply if shift is meaningful (> 0.5 × cell size)
                if np.linalg.norm(coordinate_shift) > 0.5 * min(sg_dx, sg_dy, sg_dz):
                    df[x_col] = df[x_col] + coordinate_shift[0]
                    df[y_col] = df[y_col] + coordinate_shift[1]
                    df[z_col] = df[z_col] + coordinate_shift[2]
                    logger.info(f"✅ Applied min-corner coordinate shift: {coordinate_shift}")
                else:
                    coordinate_shift = None
                    logger.debug("Coordinate shift negligible — skipping")
            except Exception as e:
                logger.warning(f"Could not compute coordinate shift from SGSIM grid: {e}")

        # ── Block dimensions ──────────────────────────────────────────
        dx = df["DX"].values[0] if "DX" in df.columns else None
        dy = df["DY"].values[0] if "DY" in df.columns else None
        dz = df["DZ"].values[0] if "DZ" in df.columns else None
        for alt_x, alt_y, alt_z in [("XINC", "YINC", "ZINC")]:
            if dx is None and alt_x in df.columns:
                dx = df[alt_x].values[0]
            if dy is None and alt_y in df.columns:
                dy = df[alt_y].values[0]
            if dz is None and alt_z in df.columns:
                dz = df[alt_z].values[0]

        x_unique = np.sort(df[x_col].unique())
        y_unique = np.sort(df[y_col].unique())
        z_unique = np.sort(df[z_col].unique())
        nx, ny, nz = len(x_unique), len(y_unique), len(z_unique)

        dx_est = np.median(np.diff(x_unique)) if len(x_unique) > 1 else 10.0
        dy_est = np.median(np.diff(y_unique)) if len(y_unique) > 1 else 10.0
        dz_est = np.median(np.diff(z_unique)) if len(z_unique) > 1 else 5.0
        dx = dx if dx is not None else dx_est
        dy = dy if dy is not None else dy_est
        dz = dz if dz is not None else dz_est

        is_regular_grid = (nx * ny * nz == len(df))
        n_blocks = len(df)
        class_col_name = "CLASS_FINAL" if "CLASS_FINAL" in df.columns else "CLASS_AUTO"
        logger.info(f"Building fallback grid: {n_blocks:,} blocks "
                     f"(DX={dx:.1f}, DY={dy:.1f}, DZ={dz:.1f}, regular={is_regular_grid})")

        if is_regular_grid:
            xmin = x_unique.min() - dx / 2
            ymin = y_unique.min() - dy / 2
            zmin = z_unique.min() - dz / 2
            xmax = x_unique.max() + dx / 2
            ymax = y_unique.max() + dy / 2
            zmax = z_unique.max() + dz / 2

            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)
            z_edges = np.linspace(zmin, zmax, nz + 1)

            grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)

            if coordinate_shift is not None:
                grid._coordinate_shifted = True

            # Map df rows → VTK cell indices (X fastest, then Y, then Z)
            xi = np.searchsorted(x_unique, df[x_col].values)
            yi = np.searchsorted(y_unique, df[y_col].values)
            zi = np.searchsorted(z_unique, df[z_col].values)
            cell_indices = xi + nx * (yi + ny * zi)

            class_ordered = np.full(nx * ny * nz, 3, dtype=np.int32)
            class_ordered[cell_indices] = class_values
            grid.cell_data["Classification"] = class_ordered
            grid.cell_data["CLASS"] = class_ordered

            cat_ordered = np.full(nx * ny * nz, "Unclassified", dtype=object)
            cat_ordered[cell_indices] = cat_labels
            grid.cell_data["Category"] = cat_ordered

            for col in ["DIST_REAL_1ST", "N_HOLES_MEAS", "N_HOLES_IND", "N_HOLES_INF"]:
                if col in df.columns:
                    v = np.full(nx * ny * nz, np.nan)
                    v[cell_indices] = df[col].values
                    grid.cell_data[col] = v

        else:
            centers = df[[x_col, y_col, z_col]].values
            hdx, hdy, hdz = dx / 2, dy / 2, dz / 2
            offsets = np.array([
                [-hdx, -hdy, -hdz], [hdx, -hdy, -hdz],
                [hdx, hdy, -hdz], [-hdx, hdy, -hdz],
                [-hdx, -hdy, hdz], [hdx, -hdy, hdz],
                [hdx, hdy, hdz], [-hdx, hdy, hdz],
            ])
            all_vertices = (centers[:, np.newaxis, :] + offsets).reshape(-1, 3)
            base_indices = np.arange(n_blocks) * 8
            cell_indices = base_indices[:, np.newaxis] + np.arange(8)
            cells = np.hstack([
                np.full((n_blocks, 1), 8, dtype=np.int64), cell_indices
            ]).flatten()
            cell_types = np.full(n_blocks, 12, dtype=np.uint8)

            grid = pv.UnstructuredGrid(cells, cell_types, all_vertices)
            grid.cell_data["Classification"] = class_values
            grid.cell_data["CLASS"] = class_values
            grid.cell_data["Category"] = cat_labels

            if coordinate_shift is not None:
                grid._coordinate_shifted = True

            for col in ["DIST_REAL_1ST", "N_HOLES_MEAS", "N_HOLES_IND", "N_HOLES_INF"]:
                if col in df.columns:
                    grid.cell_data[col] = df[col].values

        return grid

    def _export_audit_report(self):
        """Export classification audit report to CSV."""
        if not self.classification_result:
            QMessageBox.warning(self, "No Results", "Run classification first.")
            return
        
        # Get save path (CSV format)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audit Report", 
            "jorc_classification_audit.csv", 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not path:
            return
        
        try:
            result = self.classification_result
            
            # Ensure .csv extension
            if not path.lower().endswith('.csv'):
                path += '.csv'
            
            # Export classified blocks
            blocks_path = path
            result.classified_df.to_csv(blocks_path, index=False)
            logger.info(f"Exported {len(result.classified_df):,} classified blocks to {blocks_path}")
            
            # Export summary to separate file
            summary_path = path.replace('.csv', '_summary.csv')
            summary_rows = []
            for cat in CLASSIFICATION_ORDER:
                data = result.summary.get(cat, {})
                if isinstance(data, dict):
                    summary_rows.append({
                        'Category': cat,
                        'Block_Count': data.get('count', 0),
                        'Percentage': data.get('percentage', 0),
                        'Avg_Distance_m': data.get('avg_distance_m', ''),
                        'Min_Distance_m': data.get('min_distance_m', ''),
                        'Max_Distance_m': data.get('max_distance_m', ''),
                        'Color': data.get('color', CLASSIFICATION_COLORS.get(cat, ''))
                    })
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(summary_path, index=False)
            
            # Export parameters to separate file
            params_path = path.replace('.csv', '_parameters.csv')
            var = result.variogram
            rules = result.ruleset
            params_rows = [
                {'Parameter': 'Variogram_Range_Major_m', 'Value': var.range_major},
                {'Parameter': 'Variogram_Range_Semi_m', 'Value': var.range_semi},
                {'Parameter': 'Variogram_Range_Minor_m', 'Value': var.range_minor},
                {'Parameter': 'Variogram_Azimuth_deg', 'Value': var.azimuth},
                {'Parameter': 'Variogram_Dip_deg', 'Value': var.dip},
                {'Parameter': 'Variogram_Sill', 'Value': var.sill},
                {'Parameter': 'Measured_Max_Distance_pct', 'Value': f"{rules.measured.max_iso_distance * 100:.0f}%"},
                {'Parameter': 'Measured_Min_Holes', 'Value': rules.measured.min_unique_holes},
                {'Parameter': 'Indicated_Max_Distance_pct', 'Value': f"{rules.indicated.max_iso_distance * 100:.0f}%"},
                {'Parameter': 'Indicated_Min_Holes', 'Value': rules.indicated.min_unique_holes},
                {'Parameter': 'Inferred_Max_Distance_pct', 'Value': f"{rules.inferred.max_iso_distance * 100:.0f}%"},
                {'Parameter': 'Inferred_Min_Holes', 'Value': rules.inferred.min_unique_holes},
                {'Parameter': 'Domain', 'Value': result.domain_name or 'Full Model'},
                {'Parameter': 'Geology_Confidence', 'Value': rules.geology_confidence},
                {'Parameter': 'Execution_Time_s', 'Value': f"{result.execution_time_seconds:.2f}"},
                {'Parameter': 'Total_Blocks', 'Value': result.summary.get('total_blocks', len(result.classified_df))},
                {'Parameter': 'Export_Timestamp', 'Value': datetime.now().isoformat()},
            ]
            
            params_df = pd.DataFrame(params_rows)
            params_df.to_csv(params_path, index=False)
            
            QMessageBox.information(
                self, "Export Complete",
                f"Audit report exported successfully:\n\n"
                f"• Classified blocks: {blocks_path}\n"
                f"• Summary: {summary_path}\n"
                f"• Parameters: {params_path}\n\n"
                f"Total: {len(result.classified_df):,} blocks"
            )
            
            logger.info(f"Audit report exported: blocks={blocks_path}, summary={summary_path}, params={params_path}")
            
        except Exception as e:
            logger.exception("Export error")
            QMessageBox.critical(self, "Export Error", f"Failed to export audit report:\n{e}")
