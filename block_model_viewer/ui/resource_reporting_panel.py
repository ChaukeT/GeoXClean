"""
Resource Reporting Panel
========================

PyQt panel for resource reporting with real-time progress updates.

Provides configuration for:
- Density modes (constant, domain-based, per-block)
- Volume modes (field-based or constant dimensions)
- Grade field selection
- Classification field selection

Displays clean resource summary table with:
- Block counts, volumes, densities, tonnages
- Mass-weighted grades and contained metal
- Totals rows for M+I and All classifications

Author: GeoX Mining Software Platform
"""

from __future__ import annotations

import logging
import csv
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox,
    QSpinBox, QComboBox, QPushButton, QLabel, QMessageBox,
    QWidget, QFrame, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QSplitter, QScrollArea, QSizePolicy,
    QApplication, QToolButton, QCheckBox, QFileDialog, QGridLayout,
    QRadioButton, QButtonGroup, QTextEdit, QTableWidgetItem
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from PyQt6.QtGui import QColor, QFont

from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from ..models.resource_reporting_engine import (
    ResourceReportingEngine,
    DensityConfig,
    VolumeConfig,
    ResourceSummaryResult,
    ResourceSummaryRow,
)
from ..utils.coordinate_utils import ensure_xyz_columns
# TRF-012: Import filter for transformed columns
from ..models.transform import filter_transformed_columns, is_transformed_column
from .modern_styles import ModernColors, get_theme_colors

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Background Worker for Resource Reporting (Non-blocking)
# ------------------------------------------------------------------ #

class ResourceReportingWorker(QObject):
    """Worker to run resource reporting in background thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # ResourceSummaryResult
    error = pyqtSignal(str)

    def __init__(self, engine, density_config, volume_config):
        super().__init__()
        self.engine = engine
        self.density_config = density_config
        self.volume_config = volume_config

    def run(self):
        """Execute resource reporting in background."""
        try:
            result = self.engine.compute_summary(
                density_config=self.density_config,
                volume_config=self.volume_config,
                progress_callback=self._emit_progress
            )
            self.finished.emit(result)
        except Exception as e:
            logger.exception("Resource reporting worker error")
            self.error.emit(str(e))

    def _emit_progress(self, pct: int, msg: str):
        """Emit progress signal (thread-safe)."""
        self.progress.emit(pct, msg)

# ------------------------------------------------------------------ #
# Modern Slider Widget (Reusable from Classification Panel)
# ------------------------------------------------------------------ #

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
        self.accent_color = color  # Store the accent color for theme refresh

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Row 1: Label and Real Value
        top_row = QHBoxLayout()
        self.lbl_title = QLabel(label)
        top_row.addWidget(self.lbl_title)

        top_row.addStretch()

        self.lbl_real_value = QLabel("= 0.0 m")
        top_row.addWidget(self.lbl_real_value)
        layout.addLayout(top_row)

        # Row 2: Slider and Spinbox
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_pct, max_pct)
        self.slider.setValue(default_pct)
        self.slider.setCursor(Qt.CursorShape.PointingHandCursor)

        self.spin = QSpinBox()
        self.spin.setRange(min_pct, max_pct)
        self.spin.setValue(default_pct)
        self.spin.setSuffix("%")
        self.spin.setFixedWidth(80)

        # Sync controls
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._on_change)

        ctrl_row.addWidget(self.slider)
        ctrl_row.addWidget(self.spin)
        layout.addLayout(ctrl_row)

        # Init label
        self._on_change(default_pct)

        # Apply initial theme
        self.refresh_theme()

    def _on_change(self, val):
        real_dist = (val / 100.0) * self.range_major
        self.lbl_real_value.setText(f"= {real_dist:.1f} m")
        self.valueChanged.emit(val)

    def set_variogram_range(self, r):
        self.range_major = r
        self._on_change(self.slider.value())

    def value(self):
        return self.slider.value()

    def refresh_theme(self):
        """Update styles for current theme."""
        self.lbl_title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-weight: 500;")
        self.lbl_real_value.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-family: monospace; font-weight: bold;")

        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {ModernColors.ELEVATED_BG};
                height: 8px;
                border-radius: 4px;
            }}
            QSlider::sub-page:horizontal {{
                background: {self.accent_color};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {ModernColors.TEXT_PRIMARY};
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }}
        """)

        self.spin.setStyleSheet(f"""
            QSpinBox {{
                border: 1px solid {ModernColors.BORDER};
                background: {ModernColors.CARD_BG};
                color: {self.accent_color};
                font-weight: bold;
                padding: 4px;
                border-radius: 4px;
            }}
        """)

# ------------------------------------------------------------------ #
# Resource Reporting Panel
# ------------------------------------------------------------------ #

class ResourceReportingPanel(BaseAnalysisPanel):
    task_name = "resource_reporting"
    reporting_complete = pyqtSignal(object)

    def __init__(self, parent=None):
        # Data State
        self.block_model_data = None
        self.summary_result = None

        # Storage for multiple block model sources
        self._block_model_sources: Dict[str, Any] = {}
        self._available_sources: list = []
        self._current_source: str = ""

        super().__init__(parent=parent, panel_id="resource_reporting")
        self.setWindowTitle("Resource Summary (JORC/SAMREC)")

        # Build UI
        self._build_ui()

        # Apply theme after UI is built
        self.refresh_theme()

        # Registry connections will be initialized when controller is bound via bind_controller()

    def _build_ui(self):
        """Build the UI layout."""
        # --- UI Construction ---
        # 1. Header
        self.main_layout.addWidget(self._create_header())

        # 2. Splitter (The "Two Window" feel)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(4)

        # LEFT PANE: Configuration (Scrollable)
        self.left_pane = self._create_config_pane()
        self.splitter.addWidget(self.left_pane)

        # RIGHT PANE: Results (Static)
        self.right_pane = self._create_results_pane()
        self.splitter.addWidget(self.right_pane)

        # Set initial sizes (50% config, 50% results)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([750, 750])  # Equal initial sizes

        self.main_layout.addWidget(self.splitter)

        # Initialize Values
        self._on_var_changed()

    def _populate_data_combos(self, df):
        """Populate combo boxes with available columns from the block model data."""
        if df is None or df.empty:
            return

        # Get all column names
        all_columns = df.columns.tolist()

        # Populate grade combo with numeric columns (potential grade fields)
        # TRF-012 COMPLIANCE: Filter out transformed columns to prevent statistical leakage
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out transformed columns (e.g., _NS, _LOG, _BC suffixes)
        physical_grade_columns = filter_transformed_columns(numeric_columns)
        
        # Log if any transformed columns were filtered
        n_filtered = len(numeric_columns) - len(physical_grade_columns)
        if n_filtered > 0:
            logger.info(
                f"TRF-012: Filtered {n_filtered} transformed column(s) from grade selection. "
                f"Use physical grades only for resource reporting."
            )
        
        if physical_grade_columns:
            current_grade = self.grade_combo.currentText()
            self.grade_combo.blockSignals(True)
            self.grade_combo.clear()
            self.grade_combo.addItems(sorted(physical_grade_columns))
            self.grade_combo.blockSignals(False)

            # Try to restore previous selection if it exists in the new data
            if current_grade in numeric_columns:
                self.grade_combo.setCurrentText(current_grade)
            elif self.grade_combo.count() > 0:
                # Default to first available column
                self.grade_combo.setCurrentIndex(0)

        # Populate classification combo with all columns (classifications can be text or numeric codes)
        if all_columns:
            current_class = self.class_combo.currentText()
            self.class_combo.blockSignals(True)
            self.class_combo.clear()
            self.class_combo.addItems(sorted(all_columns))
            self.class_combo.blockSignals(False)

            # Try to restore previous selection if it exists in the new data
            if current_class in all_columns:
                self.class_combo.setCurrentText(current_class)
            elif "CLASS_FINAL" in all_columns:
                self.class_combo.setCurrentText("CLASS_FINAL")
            elif "CLASS" in all_columns:
                self.class_combo.setCurrentText("CLASS")
            elif self.class_combo.count() > 0:
                self.class_combo.setCurrentIndex(0)

        # Populate domain combo with all columns (domain can be numeric or text)
        if all_columns:
            current_domain = self.domain_combo.currentText()
            self.domain_combo.blockSignals(True)
            self.domain_combo.clear()
            self.domain_combo.addItem("(None)")
            self.domain_combo.addItems(sorted(all_columns))
            self.domain_combo.blockSignals(False)

            # Try to restore previous selection if it exists in the new data
            if current_domain and current_domain != "(None)" and current_domain in all_columns:
                self.domain_combo.setCurrentText(current_domain)
            elif "DOMAIN" in all_columns:
                self.domain_combo.setCurrentText("DOMAIN")
            elif "ZONE" in all_columns:
                self.domain_combo.setCurrentText("ZONE")
            elif "LITHOLOGY" in all_columns:
                self.domain_combo.setCurrentText("LITHOLOGY")
            elif "ROCK_TYPE" in all_columns:
                self.domain_combo.setCurrentText("ROCK_TYPE")
            else:
                self.domain_combo.setCurrentIndex(0)  # (None)

            # Enable/disable domain-based density based on domain selection
            self._on_domain_selection_changed()

        # Populate volume field combo with all columns
        if all_columns:
            current_volume = self.volume_field_combo.currentText()
            self.volume_field_combo.blockSignals(True)
            self.volume_field_combo.clear()
            self.volume_field_combo.addItems(sorted(all_columns))
            self.volume_field_combo.blockSignals(False)

            # Try to restore previous selection if it exists in the new data
            if current_volume in all_columns:
                self.volume_field_combo.setCurrentText(current_volume)
            elif "BLOCK_VOLUME" in all_columns:
                self.volume_field_combo.setCurrentText("BLOCK_VOLUME")
            elif self.volume_field_combo.count() > 0:
                self.volume_field_combo.setCurrentIndex(0)

        # Populate density field combo with all columns
        if all_columns:
            current_density = self.density_field_combo.currentText()
            self.density_field_combo.blockSignals(True)
            self.density_field_combo.clear()
            self.density_field_combo.addItems(sorted(all_columns))
            self.density_field_combo.blockSignals(False)

            # Try to restore previous selection if it exists in the new data
            if current_density in all_columns:
                self.density_field_combo.setCurrentText(current_density)
            elif "DENSITY" in all_columns:
                self.density_field_combo.setCurrentText("DENSITY")
            elif self.density_field_combo.count() > 0:
                self.density_field_combo.setCurrentIndex(0)

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
        """Apply theme colors to the panel."""
        self.setStyleSheet(f"""
            QWidget {{ background-color: {ModernColors.PANEL_BG}; color: {ModernColors.TEXT_PRIMARY}; font-family: 'Segoe UI', sans-serif; }}
            QGroupBox {{
                border: 1px solid {ModernColors.BORDER};
                margin-top: 20px;
                border-radius: 4px;
                padding-top: 15px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {ModernColors.ACCENT_PRIMARY};
            }}
            QScrollArea {{ border: none; background-color: transparent; }}
            QScrollBar:vertical {{
                border: none; background: {ModernColors.ELEVATED_BG}; width: 12px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {ModernColors.BORDER_LIGHT}; min-height: 20px; border-radius: 6px;
            }}
        """)

    def refresh_theme(self):
        """Refresh all widget styles when theme changes."""
        # Re-apply base theme
        self._apply_theme()

        # Update header frame
        if hasattr(self, 'header_frame'):
            self.header_frame.setStyleSheet(f"background-color: {ModernColors.ELEVATED_BG}; border-bottom: 2px solid {ModernColors.BORDER};")

        if hasattr(self, 'header_title'):
            self.header_title.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {ModernColors.TEXT_PRIMARY};")

        # Update refresh button
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.setStyleSheet(f"""
                QPushButton {{
                    background: {ModernColors.CARD_BG}; color: {ModernColors.ACCENT_PRIMARY}; border: 1px solid {ModernColors.ACCENT_PRIMARY};
                    padding: 5px 15px; border-radius: 4px; font-weight: bold;
                }}
                QPushButton:hover {{ background: {ModernColors.CARD_HOVER}; }}
            """)

        # Update classification indicator and status label
        if hasattr(self, 'classification_indicator'):
            self.classification_indicator.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10pt; font-weight: bold;")

        if hasattr(self, 'status_lbl'):
            self.status_lbl.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-weight: bold; background: {ModernColors.CARD_BG}; padding: 5px 10px; border-radius: 4px;")

        # Update splitter
        if hasattr(self, 'splitter'):
            self.splitter.setStyleSheet(f"""
                QSplitter::handle {{ background-color: {ModernColors.BORDER}; }}
                QSplitter::handle:hover {{ background-color: {ModernColors.ACCENT_PRIMARY}; }}
            """)

        # Update all combo boxes
        combo_style = f"background: {ModernColors.CARD_BG}; border: 1px solid {ModernColors.BORDER}; padding: 5px; color: {ModernColors.TEXT_PRIMARY};"
        for combo in [self.source_combo, self.class_combo, self.grade_combo, self.domain_combo,
                      self.volume_field_combo, self.density_field_combo]:
            if hasattr(self, combo.objectName()) or combo:
                combo.setStyleSheet(combo_style)

        # Update spin boxes
        spin_style = f"background: {ModernColors.CARD_BG}; border: 1px solid {ModernColors.BORDER}; padding: 5px; color: {ModernColors.TEXT_PRIMARY};"
        for spin in [self.dx_spin, self.dy_spin, self.dz_spin, self.density_spin]:
            if hasattr(self, spin.objectName()) or spin:
                spin.setStyleSheet(spin_style)

        # Update checkboxes
        if hasattr(self, 'chk_grade_percent'):
            self.chk_grade_percent.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY};")

        # Update info labels
        if hasattr(self, 'source_info_label'):
            self.source_info_label.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-style: italic; padding: 5px;")

        if hasattr(self, 'lbl_domain_status'):
            self.lbl_domain_status.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-style: italic; font-size: 9pt; padding: 5px;")

        if hasattr(self, 'progress_lbl'):
            self.progress_lbl.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-size: 10pt;")

        # Update buttons
        primary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.ACCENT_PRIMARY}; color: white; border: none;
                padding: 10px 20px; border-radius: 4px; font-weight: bold; font-size: 12pt;
            }}
            QPushButton:hover {{ background: {ModernColors.ACCENT_HOVER}; }}
            QPushButton:disabled {{ background: {ModernColors.BORDER}; color: {ModernColors.TEXT_DISABLED}; }}
        """

        secondary_btn_style = f"""
            QPushButton {{
                background: {ModernColors.CARD_BG}; color: {ModernColors.TEXT_PRIMARY}; border: 1px solid {ModernColors.BORDER};
                padding: 8px 15px; border-radius: 4px; font-weight: 500;
            }}
            QPushButton:hover {{ background: {ModernColors.CARD_HOVER}; border-color: {ModernColors.BORDER_LIGHT}; }}
            QPushButton:disabled {{ background: {ModernColors.PANEL_BG}; color: {ModernColors.TEXT_DISABLED}; }}
        """

        if hasattr(self, 'btn_run'):
            self.btn_run.setStyleSheet(primary_btn_style)

        for btn in [self.btn_export, self.btn_export_excel, self.btn_generate_statement, self.btn_populate_domains]:
            if hasattr(self, btn.objectName()) or btn:
                btn.setStyleSheet(secondary_btn_style)

        # Update progress bar
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid {ModernColors.BORDER};
                    border-radius: 6px;
                    text-align: center;
                    background: {ModernColors.CARD_BG};
                    color: {ModernColors.TEXT_PRIMARY};
                    font-weight: bold;
                }}
                QProgressBar::chunk {{
                    background-color: {ModernColors.ACCENT_PRIMARY};
                    border-radius: 6px;
                }}
            """)

        # Update table
        if hasattr(self, 'table'):
            self.table.setStyleSheet(f"""
                QTableWidget {{
                    background-color: {ModernColors.CARD_BG};
                    alternate-background-color: {ModernColors.ELEVATED_BG};
                    gridline-color: {ModernColors.BORDER};
                    border: 1px solid {ModernColors.BORDER};
                    border-radius: 4px;
                    color: {ModernColors.TEXT_PRIMARY};
                }}
                QTableWidget::item {{
                    padding: 5px;
                    border-bottom: 1px solid {ModernColors.BORDER};
                }}
                QTableWidget::item:selected {{
                    background-color: {ModernColors.ACCENT_PRIMARY};
                    color: white;
                }}
                QHeaderView::section {{
                    background-color: {ModernColors.ELEVATED_BG};
                    color: {ModernColors.TEXT_PRIMARY};
                    padding: 6px;
                    border: none;
                    border-bottom: 2px solid {ModernColors.ACCENT_PRIMARY};
                    border-right: 1px solid {ModernColors.BORDER};
                    font-weight: bold;
                }}
            """)

        # Update domain density table if exists
        if hasattr(self, 'domain_density_table'):
            self.domain_density_table.setStyleSheet(f"""
                QTableWidget {{
                    background-color: {ModernColors.CARD_BG};
                    gridline-color: {ModernColors.BORDER};
                    border: 1px solid {ModernColors.BORDER};
                    border-radius: 4px;
                    color: {ModernColors.TEXT_PRIMARY};
                }}
                QTableWidget::item {{
                    padding: 4px;
                }}
                QHeaderView::section {{
                    background-color: {ModernColors.ELEVATED_BG};
                    color: {ModernColors.TEXT_PRIMARY};
                    padding: 5px;
                    border: 1px solid {ModernColors.BORDER};
                    font-weight: bold;
                }}
            """)

        # Update results title
        if hasattr(self, 'results_title'):
            self.results_title.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {ModernColors.ACCENT_PRIMARY}; margin-bottom: 10px;")

    def _create_header(self) -> QWidget:
        frame = QFrame()
        frame.setFixedHeight(60)
        self.header_frame = frame  # Store for theme refresh
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Resource Summary Manager")
        self.header_title = title  # Store for theme refresh
        layout.addWidget(title)

        layout.addStretch()

        # Refresh button to reload data from registry
        self.refresh_btn = QPushButton("⟳ Refresh")
        self.refresh_btn.setToolTip("Reload data from registry (Block Model, Drillholes)")
        self.refresh_btn.clicked.connect(self._refresh_data)
        layout.addWidget(self.refresh_btn)

        layout.addSpacing(20)

        # Status container with classification indicator
        status_container = QWidget()
        status_layout = QHBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(5)

        self.classification_indicator = QLabel("")
        status_layout.addWidget(self.classification_indicator)

        self.status_lbl = QLabel("WAITING FOR DATA")
        status_layout.addWidget(self.status_lbl)

        layout.addWidget(status_container)

        return frame

    def _create_config_pane(self) -> QWidget:
        """Creates the scrollable left pane."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # 0. Block Model Source Selection
        source_group = QGroupBox("Block Model Source")
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(8)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Select Block Model:"))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(200)
        self.source_combo.addItem("No block model loaded", "none")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_combo.setToolTip("Select which block model to use for resource reporting")
        source_row.addWidget(self.source_combo, stretch=1)
        source_layout.addLayout(source_row)

        self.source_info_label = QLabel("No block model selected")
        source_layout.addWidget(self.source_info_label)

        layout.addWidget(source_group)

        # 1. Data/Field Selection Group
        data_group = QGroupBox("1. Data Selection")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(8)

        # Classification field
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Classification Field:"))
        self.class_combo = QComboBox()
        # Combo will be populated dynamically when data is loaded
        class_layout.addWidget(self.class_combo)
        data_layout.addLayout(class_layout)

        # Grade field
        grade_layout = QHBoxLayout()
        grade_layout.addWidget(QLabel("Grade Field:"))
        self.grade_combo = QComboBox()
        # Combo will be populated dynamically when data is loaded
        grade_layout.addWidget(self.grade_combo)

        # Grade units checkbox
        self.chk_grade_percent = QCheckBox("Grade stored as %")
        self.chk_grade_percent.setChecked(True)  # Default to percent
        self.chk_grade_percent.setToolTip("Check if grade values are stored as percentages (0-100).\nUncheck if stored as fractions (0-1).")
        grade_layout.addWidget(self.chk_grade_percent)

        data_layout.addLayout(grade_layout)

        # Domain field (optional)
        domain_layout = QHBoxLayout()
        domain_layout.addWidget(QLabel("Domain Field:"))
        self.domain_combo = QComboBox()
        self.domain_combo.setToolTip("Optional. Select a domain column for domain-based density assignment.\nLeave as '(None)' if not needed.")
        # Combo will be populated dynamically when data is loaded
        domain_layout.addWidget(self.domain_combo)
        data_layout.addLayout(domain_layout)

        layout.addWidget(data_group)

        # 2. Volume Configuration Group
        volume_group = QGroupBox("2. Volume Configuration")
        volume_layout = QVBoxLayout(volume_group)
        volume_layout.setSpacing(8)

        # Volume mode radio buttons
        self.volume_mode_group = QButtonGroup(volume_group)

        # Field mode
        self.rb_volume_field = QRadioButton("Use existing volume field")
        self.rb_volume_field.setChecked(True)
        self.volume_mode_group.addButton(self.rb_volume_field, 0)
        volume_layout.addWidget(self.rb_volume_field)

        field_layout = QHBoxLayout()
        field_layout.addSpacing(12)
        field_layout.addWidget(QLabel("Field name:"))
        self.volume_field_combo = QComboBox()
        # Combo will be populated dynamically when data is loaded
        field_layout.addWidget(self.volume_field_combo)
        volume_layout.addLayout(field_layout)

        # Constant mode
        self.rb_volume_constant = QRadioButton("Use constant block dimensions")
        self.volume_mode_group.addButton(self.rb_volume_constant, 1)
        volume_layout.addWidget(self.rb_volume_constant)

        dims_layout = QHBoxLayout()
        dims_layout.addSpacing(12)
        dims_layout.addWidget(QLabel("X:"))
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 1000.0)
        self.dx_spin.setValue(10.0)
        self.dx_spin.setSuffix(" m")
        dims_layout.addWidget(self.dx_spin)

        dims_layout.addWidget(QLabel("Y:"))
        self.dy_spin = QDoubleSpinBox()
        self.dy_spin.setRange(0.1, 1000.0)
        self.dy_spin.setValue(10.0)
        self.dy_spin.setSuffix(" m")
        dims_layout.addWidget(self.dy_spin)

        dims_layout.addWidget(QLabel("Z:"))
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 1000.0)
        self.dz_spin.setValue(5.0)
        self.dz_spin.setSuffix(" m")
        dims_layout.addWidget(self.dz_spin)

        volume_layout.addLayout(dims_layout)

        # Enable/disable controls based on radio buttons
        self.rb_volume_field.toggled.connect(self._on_volume_mode_changed)
        self._on_volume_mode_changed()

        layout.addWidget(volume_group)

        # 3. Density Configuration Group
        density_group = QGroupBox("3. Density Configuration")
        density_layout = QVBoxLayout(density_group)
        density_layout.setSpacing(8)

        # Density mode radio buttons
        self.density_mode_group = QButtonGroup(density_group)

        # Constant density
        self.rb_density_constant = QRadioButton("Constant density")
        self.rb_density_constant.setToolTip("Apply the same density value to all blocks in the model.")
        self.rb_density_constant.setChecked(True)
        self.density_mode_group.addButton(self.rb_density_constant, 0)
        density_layout.addWidget(self.rb_density_constant)

        const_layout = QHBoxLayout()
        const_layout.addSpacing(12)
        const_layout.addWidget(QLabel("Density:"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 10.0)
        self.density_spin.setValue(2.8)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setSuffix(" t/m³")
        const_layout.addWidget(self.density_spin)
        density_layout.addLayout(const_layout)

        # Domain density
        self.rb_density_domain = QRadioButton("Density per domain")
        self.rb_density_domain.setToolTip("Assign different densities to different geological domains.\nUse 'Populate from Block Model' to load domains automatically.\nRequires a Domain Field to be selected in Data Selection.")
        self.density_mode_group.addButton(self.rb_density_domain, 1)
        density_layout.addWidget(self.rb_density_domain)

        # Domain density table widget
        self.domain_density_widget = QWidget()
        domain_table_layout = QVBoxLayout(self.domain_density_widget)
        domain_table_layout.setContentsMargins(12, 5, 0, 5)

        # Header with populate button
        domain_header_layout = QHBoxLayout()
        domain_header_layout.addWidget(QLabel("Domain Densities:"))
        domain_header_layout.addStretch()

        self.btn_populate_domains = QPushButton("Populate from Block Model")
        self.btn_populate_domains.clicked.connect(self._populate_domain_densities)
        domain_header_layout.addWidget(self.btn_populate_domains)

        domain_table_layout.addLayout(domain_header_layout)

        # Domain density table
        self.domain_density_table = QTableWidget()
        self.domain_density_table.setColumnCount(2)
        self.domain_density_table.setHorizontalHeaderLabels(["Domain", "Density (t/m³)"])
        self.domain_density_table.setMaximumHeight(150)

        # Set column widths
        header = self.domain_density_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.domain_density_table.setColumnWidth(1, 100)

        domain_table_layout.addWidget(self.domain_density_table)

        # Status label for domain density
        self.lbl_domain_status = QLabel("No domains loaded - use 'Populate from Block Model' button")
        domain_table_layout.addWidget(self.lbl_domain_status)

        density_layout.addWidget(self.domain_density_widget)

        # Block density
        self.rb_density_block = QRadioButton("Density from block model column")
        self.rb_density_block.setToolTip("Use per-block density values from a column in the block model.\nSelect the column name containing density values.")
        self.density_mode_group.addButton(self.rb_density_block, 2)
        density_layout.addWidget(self.rb_density_block)

        block_layout = QHBoxLayout()
        block_layout.addSpacing(12)
        block_layout.addWidget(QLabel("Field name:"))
        self.density_field_combo = QComboBox()
        # Combo will be populated dynamically when data is loaded
        block_layout.addWidget(self.density_field_combo)
        density_layout.addLayout(block_layout)

        # Enable/disable controls based on radio buttons
        self.rb_density_constant.toggled.connect(self._on_density_mode_changed)
        self.rb_density_domain.toggled.connect(self._on_density_mode_changed)
        self.domain_combo.currentTextChanged.connect(self._on_domain_selection_changed)
        self.rb_density_block.toggled.connect(self._on_density_mode_changed)
        self._on_density_mode_changed()

        # Initially hide domain density widget
        self.domain_density_widget.setVisible(False)

        layout.addWidget(density_group)

        # 4. Control Buttons
        button_layout = QHBoxLayout()

        self.btn_run = QPushButton("🔍 RUN RESOURCE SUMMARY")
        self.btn_run.clicked.connect(self.run_resource_reporting)
        button_layout.addWidget(self.btn_run)

        button_layout.addStretch()

        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_export.setEnabled(False)
        button_layout.addWidget(self.btn_export)

        self.btn_export_excel = QPushButton("📊 Export Excel")
        self.btn_export_excel.clicked.connect(self._export_excel)
        self.btn_export_excel.setEnabled(False)
        button_layout.addWidget(self.btn_export_excel)

        self.btn_generate_statement = QPushButton("📄 Generate Statement")
        self.btn_generate_statement.clicked.connect(self._generate_statement)
        self.btn_generate_statement.setEnabled(False)
        button_layout.addWidget(self.btn_generate_statement)

        layout.addLayout(button_layout)

        # 5. Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_lbl = QLabel("Ready to run resource summary...")
        progress_layout.addWidget(self.progress_lbl)

        layout.addWidget(progress_group)

        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_results_pane(self) -> QWidget:
        """Creates the right pane with results table."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)

        # Results header
        results_title = QLabel("Resource Summary Results")
        self.results_title = results_title  # Store for theme refresh
        layout.addWidget(results_title)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Classification", "Blocks", "Volume (m³)", "Density (t/m³)",
            "Tonnes (t)", "Grade (%)", "Contained Metal (t)"
        ])

        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 120)  # Classification
        self.table.setColumnWidth(1, 80)   # Blocks
        self.table.setColumnWidth(2, 100)  # Volume
        self.table.setColumnWidth(3, 100)  # Density
        self.table.setColumnWidth(4, 100)  # Tonnes
        self.table.setColumnWidth(5, 80)   # Grade
        self.table.setColumnWidth(6, 130)  # Contained Metal

        self.table.setAlternatingRowColors(True)

        layout.addWidget(self.table)

        layout.addStretch()

        return container

    def _on_volume_mode_changed(self):
        """Enable/disable volume controls based on mode."""
        field_mode = self.rb_volume_field.isChecked()
        self.volume_field_combo.setEnabled(field_mode)
        self.dx_spin.setEnabled(not field_mode)
        self.dy_spin.setEnabled(not field_mode)
        self.dz_spin.setEnabled(not field_mode)

    def _populate_domain_densities(self):
        """Populate domain density table from current block model."""
        if self.block_model_data is None or self.block_model_data.empty:
            QMessageBox.warning(self, "No Data", "No block model loaded to extract domains from.")
            return

        domain_col = self.domain_combo.currentText()
        if domain_col == "(None)":
            QMessageBox.warning(self, "No Domain Selected",
                "Please select a domain field in the Data Selection section\n"
                "before populating domain densities.")
            return
        if domain_col not in self.block_model_data.columns:
            QMessageBox.warning(self, "Domain Column Missing",
                f"Domain column '{domain_col}' not found in block model.\n\n"
                f"Available columns: {', '.join(self.block_model_data.columns[:10])}{'...' if len(self.block_model_data.columns) > 10 else ''}")
            return

        # Get unique domains
        unique_domains = sorted(self.block_model_data[domain_col].dropna().unique())
        if len(unique_domains) == 0:
            QMessageBox.warning(self, "No Domains", "No valid domains found in the selected column.")
            return

        # Clear existing table
        self.domain_density_table.setRowCount(0)

        # Populate table with domains
        for domain in unique_domains:
            row_idx = self.domain_density_table.rowCount()
            self.domain_density_table.insertRow(row_idx)

            # Domain name
            domain_item = QTableWidgetItem(str(domain))
            domain_item.setFlags(domain_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Read-only
            self.domain_density_table.setItem(row_idx, 0, domain_item)

            # Density input (default to 2.8)
            density_item = QTableWidgetItem("2.8")
            self.domain_density_table.setItem(row_idx, 1, density_item)

        # Update status
        self.lbl_domain_status.setText(f"Loaded {len(unique_domains)} domains - enter density values below")
        self.lbl_domain_status.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-style: italic; font-size: 9pt; padding: 5px;")

        logger.info(f"Populated domain density table with {len(unique_domains)} domains")
        QMessageBox.information(self, "Domains Populated",
            f"Loaded {len(unique_domains)} domains from block model.\n\n"
            "Please enter density values for each domain.")

    def _on_density_mode_changed(self):
        """Enable/disable density controls based on mode."""
        constant_mode = self.rb_density_constant.isChecked()
        domain_mode = self.rb_density_domain.isChecked()
        block_mode = self.rb_density_block.isChecked()

        self.density_spin.setEnabled(constant_mode)
        self.domain_density_widget.setVisible(domain_mode)
        self.density_field_combo.setEnabled(block_mode)

    def _on_domain_selection_changed(self, text=None):
        """Enable/disable domain-based density option based on domain field selection."""
        domain_selected = self.domain_combo.currentText() != "(None)"
        self.rb_density_domain.setEnabled(domain_selected)
        if not domain_selected and self.rb_density_domain.isChecked():
            self.rb_density_constant.setChecked(True)

    def _on_var_changed(self):
        """Handle variogram changes (placeholder for now)."""
        pass

    def bind_controller(self, controller):
        """Override to re-initialize registry connections after controller binding."""
        super().bind_controller(controller)
        # Re-initialize registry connections now that controller is bound
        self._init_registry()

    def _init_registry(self):
        """Initialize registry connections."""
        try:
            self.reg = self.get_registry()
            if self.reg:
                connected_signals = []

                # Helper to safely connect signals
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

                # Block model data - connect to ALL sources
                safe_connect('blockModelClassified', self._on_bm_loaded)
                safe_connect('blockModelGenerated', self._on_bm_generated)
                safe_connect('blockModelLoaded', self._on_bm_generated)

                # SGSIM results - direct access without requiring classification
                safe_connect('sgsimResultsLoaded', self._on_sgsim_loaded)

                # Trigger initial load
                QTimer.singleShot(500, self._load_existing)

                logger.info(f"Resource Reporting: Connected {len(connected_signals)} signals: {connected_signals}")
        except Exception as e:
            logger.warning(f"Resource Reporting: Failed to connect to registry: {e}", exc_info=True)

    def _on_bm_loaded(self, df):
        """Handle block model loaded signal."""
        logger.info(f"Resource Reporting: Received block model ({len(df) if df is not None else 0} blocks)")

        # Use the data provided by the signal directly to avoid triggering registry access
        if df is not None and not df.empty:
            # Check if this is the same data we already have
            if self.block_model_data is not None and len(self.block_model_data) == len(df):
                logger.debug("Resource Reporting: Block model data unchanged, skipping reload")
                return

            self.block_model_data = df

            # Populate combo boxes with available columns from the block model
            self._populate_data_combos(df)

            self.classification_indicator.setText("🟢 CLASSIFIED")
            self.classification_indicator.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-size: 10pt; font-weight: bold;")
            self.status_lbl.setText(f"LOADED: {len(df):,} blocks")
        else:
            # If signal sent None/empty, trigger a full reload to check registry state
            self._load_existing()

    def _on_bm_generated(self, bm):
        """Handle block model generated/loaded signal (unclassified)."""
        try:
            import pandas as pd
            from ..models.block_model import BlockModel

            df = None
            if isinstance(bm, pd.DataFrame):
                df = bm
            elif isinstance(bm, BlockModel):
                df = bm.to_dataframe() if hasattr(bm, 'to_dataframe') else bm.data
            elif isinstance(bm, dict):
                df = bm.get('data') or bm.get('df') or bm.get('block_model')
                if isinstance(df, BlockModel):
                    df = df.to_dataframe() if hasattr(df, 'to_dataframe') else df.data

            if df is not None and not df.empty:
                logger.info(f"Resource Reporting: Received generated block model ({len(df)} blocks)")
                self.block_model_data = df
                self._populate_data_combos(df)
                self.classification_indicator.setText("⚪ UNCLASSIFIED")
                self.classification_indicator.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10pt; font-weight: bold;")
                self.status_lbl.setText(f"LOADED: {len(df):,} blocks")
        except Exception as e:
            logger.warning(f"Resource Reporting: Failed to load generated block model: {e}")

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources."""
        if results is None:
            return
        self._register_sgsim_sources(results)
        self._update_source_selector()

    def _register_source(self, name: str, data, auto_select: bool = False):
        """Register a block model source."""
        import pandas as pd

        if data is None:
            return

        # Convert to DataFrame if needed
        if hasattr(data, 'to_dataframe'):
            df = data.to_dataframe()
        elif isinstance(data, pd.DataFrame):
            df = data
        elif hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
            df = data.data
        else:
            return

        if df is None or df.empty:
            return

        self._block_model_sources[name] = df
        if name not in self._available_sources:
            self._available_sources.append(name)

        logger.info(f"Resource Reporting: Registered source '{name}' ({len(df):,} blocks)")

        if auto_select and (not self._current_source or self._current_source == ""):
            self._current_source = name
            self.block_model_data = df
            self._populate_data_combos(df)

    def _register_sgsim_sources(self, sgsim_results):
        """Register SGSIM results as multiple block model sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        import pandas as pd
        import numpy as np
        import pyvista as pv

        if sgsim_results is None:
            return

        if not isinstance(sgsim_results, dict):
            logger.warning(f"Resource Reporting: SGSIM results is not a dict, type={type(sgsim_results)}")
            return

        variable = sgsim_results.get('variable', 'Grade')
        summary = sgsim_results.get('summary', {})
        params = sgsim_results.get('params')
        grid = sgsim_results.get('grid') or sgsim_results.get('pyvista_grid')

        logger.info(f"Resource Reporting: SGSIM results keys: {list(sgsim_results.keys())}")
        logger.info(f"Resource Reporting: Summary keys: {list(summary.keys()) if summary else 'None'}")
        logger.info(f"Resource Reporting: params = {params is not None}")

        # Extract coordinates from grid or generate from params
        base_df = None
        n_blocks = 0

        if grid is not None and isinstance(grid, (pv.RectilinearGrid, pv.UnstructuredGrid, pv.StructuredGrid, pv.ImageData)):
            if hasattr(grid, 'cell_centers'):
                centers = grid.cell_centers()
                if hasattr(centers, 'points'):
                    coords = centers.points
                    base_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2]})
                    n_blocks = len(base_df)
                    logger.info(f"Resource Reporting: Extracted {n_blocks:,} cell centers from grid")

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
                n_blocks = len(base_df)
                logger.info(f"Resource Reporting: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
            except Exception as e:
                logger.warning(f"Resource Reporting: Failed to generate coords from params: {e}")

        if base_df is None or base_df.empty:
            # Fallback: realizations array with grid coordinates
            reals = sgsim_results.get('realizations') or sgsim_results.get('realizations_raw')
            if reals is not None:
                grid_x, grid_y, grid_z = sgsim_results.get('grid_x'), sgsim_results.get('grid_y'), sgsim_results.get('grid_z')
                if grid_x is not None and isinstance(reals, np.ndarray):
                    mean_estimate = np.mean(reals, axis=0) if reals.ndim == 2 else reals.ravel()
                    df = pd.DataFrame({
                        'X': np.asarray(grid_x).ravel(), 'Y': np.asarray(grid_y).ravel(),
                        'Z': np.asarray(grid_z).ravel(), variable: mean_estimate
                    })
                    self._register_source(f"SGSIM E-type Mean ({variable})", df, auto_select=False)
                    logger.info(f"Resource Reporting: Registered SGSIM E-type Mean from fallback")
            else:
                logger.warning("Resource Reporting: No grid, params, or realizations found in SGSIM results")
            return

        found_stats = []

        # Extract individual statistics from 'summary' dict
        # SGSIM stores: summary['mean'], summary['std'], summary['p10'], summary['p50'], summary['p90']
        stat_mapping = {
            'mean': 'SGSIM Mean',
            'std': 'SGSIM Std Dev',
            'p10': 'SGSIM P10',
            'p50': 'SGSIM P50',
            'p90': 'SGSIM P90',
        }

        for stat_key, display_prefix in stat_mapping.items():
            stat_data = summary.get(stat_key)
            if stat_data is not None:
                stat_values = np.asarray(stat_data).flatten()
                if len(stat_values) == n_blocks:
                    df = base_df.copy()
                    prop_name = f"{variable}_{stat_key.upper()}"
                    df[prop_name] = stat_values

                    display_name = f"{display_prefix} ({variable})"
                    self._register_source(display_name, df, auto_select=False)
                    found_stats.append(stat_key)
                    logger.info(f"Resource Reporting: Registered {display_prefix} ({variable})")

        # Also extract from grid cell_data (e.g., FE_SGSIM_MEAN)
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

                if 'MEAN' in prop_upper or 'E_TYPE' in prop_upper:
                    display_name = f"SGSIM Mean ({variable})"
                elif 'PROB' in prop_upper:
                    display_name = f"SGSIM Probability ({prop_name})"
                else:
                    display_name = f"SGSIM {prop_name}"

                self._register_source(display_name, df, auto_select=False)
                found_stats.append(prop_name)

        if found_stats:
            logger.info(f"Resource Reporting: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

    def _update_source_selector(self):
        """Update the source selector combo box."""
        if not hasattr(self, 'source_combo'):
            return

        self.source_combo.blockSignals(True)
        self.source_combo.clear()

        if not self._available_sources:
            self.source_combo.addItem("No block model loaded", "none")
            self.source_info_label.setText("Load a block model or run SGSIM simulation")
        else:
            for source_name in self._available_sources:
                df = self._block_model_sources.get(source_name)
                block_count = len(df) if df is not None else 0
                display_text = f"{source_name} ({block_count:,} blocks)"
                self.source_combo.addItem(display_text, source_name)

            if self._current_source and self._current_source in self._available_sources:
                idx = self._available_sources.index(self._current_source)
                self.source_combo.setCurrentIndex(idx)

        self.source_combo.blockSignals(False)

    def _on_source_changed(self, index: int):
        """Handle block model source selection change."""
        if index < 0 or not hasattr(self, 'source_combo'):
            return

        source_name = self.source_combo.itemData(index)
        if source_name is None or source_name == "none":
            self.block_model_data = None
            self._current_source = ""
            self.source_info_label.setText("No block model selected")
            self.status_lbl.setText("NO DATA")
            return

        if source_name in self._block_model_sources:
            self._current_source = source_name
            self.block_model_data = self._block_model_sources[source_name]
            df = self.block_model_data

            cols = [c for c in df.columns if c.upper() not in ('X', 'Y', 'Z')]
            self.source_info_label.setText(f"Properties: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")

            self._populate_data_combos(df)
            self.status_lbl.setText(f"LOADED: {len(df):,} blocks")

            # Update indicator based on source type
            if 'SGSIM' in source_name.upper():
                self.classification_indicator.setText("🔵 SGSIM")
                self.classification_indicator.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10pt; font-weight: bold;")
            elif 'CLASSIFIED' in source_name.upper():
                self.classification_indicator.setText("🟢 CLASSIFIED")
                self.classification_indicator.setStyleSheet(f"color: {ModernColors.ACCENT_PRIMARY}; font-size: 10pt; font-weight: bold;")
            else:
                self.classification_indicator.setText("⚪ UNCLASSIFIED")
                self.classification_indicator.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10pt; font-weight: bold;")

            logger.info(f"Resource Reporting: Switched to '{source_name}' ({len(df):,} blocks)")

    def _refresh_data(self):
        """Manual refresh button handler."""
        logger.info("Resource Reporting: Manual refresh requested")

        # Clear existing sources
        self._block_model_sources.clear()
        self._available_sources.clear()
        self._current_source = ""

        self._load_existing()

        # Show feedback
        source_count = len(self._available_sources)
        if source_count > 0:
            source_list = "\n".join([f"  • {s}" for s in self._available_sources[:5]])
            if source_count > 5:
                source_list += f"\n  ... and {source_count - 5} more"
            QMessageBox.information(self, "Refresh Complete",
                f"Data refreshed successfully!\n\n"
                f"Block Model Sources: {source_count}\n{source_list}")
        else:
            QMessageBox.warning(self, "No Data Found",
                "No block model data found in registry.\n\n"
                "Please run Resource Classification, SGSIM, or load a block model.")

    def _load_existing(self):
        """Load existing data from registry - registers ALL available sources."""
        if not hasattr(self, 'reg') or self.reg is None:
            logger.warning("Resource Reporting: Registry not available")
            return

        logger.info("Resource Reporting: Loading existing data from registry...")
        found_any = False

        try:
            # 1. Try to get classified block model
            bm = self.reg.get_classified_block_model()
            if bm is not None and not bm.empty:
                self._register_source("Classified Block Model", bm, auto_select=True)
                found_any = True

            # 2. Try to get regular block model
            bm = self.reg.get_block_model()
            if bm is not None and not bm.empty:
                self._register_source("Block Model (Loaded/Generated)", bm, auto_select=not found_any)
                found_any = True

            # 3. Try to get SGSIM results (registers multiple sources)
            if hasattr(self.reg, 'get_sgsim_results'):
                sgsim = self.reg.get_sgsim_results()
                if sgsim is not None:
                    self._register_sgsim_sources(sgsim)
                    if self._available_sources:
                        found_any = True

            # 4. Try to get Kriging results
            if hasattr(self.reg, 'get_kriging_results'):
                kriging = self.reg.get_kriging_results()
                if kriging is not None:
                    import pyvista as pv
                    grid = kriging.get('grid') or kriging.get('pyvista_grid')
                    if grid is not None and isinstance(grid, (pv.RectilinearGrid, pv.UnstructuredGrid, pv.StructuredGrid)):
                        if hasattr(grid, 'cell_centers'):
                            centers = grid.cell_centers()
                            if hasattr(centers, 'points'):
                                coords = centers.points
                                base_df = pd.DataFrame({'X': coords[:, 0], 'Y': coords[:, 1], 'Z': coords[:, 2]})
                                for prop_name in grid.cell_data.keys():
                                    df = base_df.copy()
                                    df[prop_name] = grid.cell_data[prop_name]
                                    self._register_source(f"Kriging: {prop_name}", df, auto_select=not found_any)
                                    found_any = True

        except Exception as e:
            logger.warning(f"Resource Reporting: Error loading sources: {e}", exc_info=True)

        # Update source selector UI
        self._update_source_selector()

        if not found_any:
            self.block_model_data = None
            self.status_lbl.setText("NO DATA FOUND")
            logger.warning("Resource Reporting: No block model data available")
        else:
            logger.info(f"Resource Reporting: Loaded {len(self._available_sources)} block model sources")

    def run_resource_reporting(self):
        """Run resource reporting in background thread."""
        if self.block_model_data is None:
            QMessageBox.warning(self, "No Data", "No block model data loaded. Please refresh or load data first.")
            return

        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_generate_statement.setEnabled(False)
        self.progress_lbl.setText("Initializing resource reporting...")
        self.progress_bar.setValue(0)

        try:
            # Validate grade field exists
            grade_col = self.grade_combo.currentText()
            grade_is_pct = self.chk_grade_percent.isChecked()

            if not grade_col or grade_col not in self.block_model_data.columns:
                QMessageBox.warning(self, "Grade Field Missing",
                    f"Grade field '{grade_col}' not found in block model.\n\n"
                    "Please select a valid grade field.")
                self.btn_run.setEnabled(True)
                return

            # Validate grade field is numeric
            grade_values = self.block_model_data[grade_col].dropna()
            if not pd.api.types.is_numeric_dtype(grade_values):
                QMessageBox.warning(self, "Grade Field Not Numeric",
                    f"Grade field '{grade_col}' does not contain numeric values.\n\n"
                    "Please select a numeric grade field.")
                self.btn_run.setEnabled(True)
                return

            if len(grade_values) > 0:
                try:
                    max_grade = float(grade_values.max())
                    min_grade = float(grade_values.min())
                except (ValueError, TypeError):
                    max_grade = 0
                    min_grade = 0

                # Check for grade units mismatch
                if grade_is_pct:
                    # If user says percent, but all values are < 1, likely wrong
                    if max_grade < 1.0 and min_grade >= 0:
                        reply = QMessageBox.question(
                            self, "Grade Units Validation",
                            f"Grade column '{grade_col}' has values ranging from {min_grade:.3f} to {max_grade:.3f}.\n\n"
                            f"You selected 'Grade stored as %' but values appear to be fractions (0-1).\n"
                            f"This will make contained metal 100× too low!\n\n"
                            f"Do you want to change to 'fractional' units?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.chk_grade_percent.setChecked(False)
                            grade_is_pct = False
                else:
                    # If user says fraction, but values are > 1, likely wrong
                    if max_grade > 1.0:
                        reply = QMessageBox.question(
                            self, "Grade Units Validation",
                            f"Grade column '{grade_col}' has values up to {max_grade:.1f}.\n\n"
                            f"You selected 'fractional' units but values appear to be percentages.\n"
                            f"This will make contained metal 100× too low!\n\n"
                            f"Do you want to change to 'percent' units?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes
                        )
                        if reply == QMessageBox.StandardButton.Yes:
                            self.chk_grade_percent.setChecked(True)
                            grade_is_pct = True

            # Validate classification field exists
            class_col = self.class_combo.currentText()
            if not class_col or class_col not in self.block_model_data.columns:
                QMessageBox.warning(self, "Classification Field Missing",
                    f"Classification field '{class_col}' not found in block model.\n\n"
                    "Please select a valid classification field.")
                self.btn_run.setEnabled(True)
                return

            # Validate classification field has reasonable values
            # Check if classification column looks like grades (too many unique numeric values)
            class_values = self.block_model_data[class_col].dropna()
            unique_count = class_values.nunique()

            # If it's numeric and has too many unique values, it might be grades
            is_numeric = pd.api.types.is_numeric_dtype(class_values)
            if is_numeric and unique_count > 10:  # Arbitrary threshold
                QMessageBox.warning(
                    self, "Classification Field Warning",
                    f"Classification column '{class_col}' has {unique_count} unique numeric values.\n\n"
                    f"This looks like grade data, not classification categories!\n\n"
                    f"Classification should be categorical (Measured/Indicated/Inferred) or integer codes.\n"
                    f"Using grade values as classes will produce meaningless summaries.\n\n"
                    f"Please select a proper classification field."
                )
                self.btn_run.setEnabled(True)
                return

            # Check for reasonable classification values
            if unique_count < 2:
                QMessageBox.warning(
                    self, "Classification Field Warning",
                    f"Classification column '{class_col}' has only {unique_count} unique value(s).\n\n"
                    f"Classification needs at least 2 categories (e.g., Measured + Indicated).\n\n"
                    f"Please select a field with proper classification categories."
                )
                self.btn_run.setEnabled(True)
                return

            # Log classification summary for user awareness
            value_counts = class_values.value_counts().head(10)
            logger.info(f"Classification field '{class_col}' has {unique_count} categories: {dict(value_counts)}")

            # Validate domain density if selected
            if self.rb_density_domain.isChecked() and self.domain_density_table.rowCount() == 0:
                QMessageBox.warning(self, "Domain Density Required",
                    "Domain density mode selected but no domains populated.\n\n"
                    "Please click 'Populate from Block Model' to load domains and enter density values.")
                self.btn_run.setEnabled(True)
                return

            # Validate density field if using block density mode
            if self.rb_density_block.isChecked():
                density_field = self.density_field_combo.currentText()
                if not density_field or density_field not in self.block_model_data.columns:
                    QMessageBox.warning(self, "Density Field Missing",
                        f"Density field '{density_field}' not found in block model.\n\n"
                        "Please select a valid density field or use constant/domain density mode.")
                    self.btn_run.setEnabled(True)
                    return

                # Validate density field is numeric
                density_values = self.block_model_data[density_field].dropna()
                if not pd.api.types.is_numeric_dtype(density_values):
                    QMessageBox.warning(self, "Density Field Not Numeric",
                        f"Density field '{density_field}' does not contain numeric values.\n\n"
                        "Density must be a numeric field (t/m³). Please select a valid density field.")
                    self.btn_run.setEnabled(True)
                    return

                # Check for reasonable density values
                if len(density_values) > 0:
                    try:
                        min_den = float(density_values.min())
                        max_den = float(density_values.max())
                        if min_den <= 0:
                            reply = QMessageBox.question(
                                self, "Density Values Warning",
                                f"Density field '{density_field}' contains non-positive values (min: {min_den:.2f}).\n\n"
                                f"Density should be positive (typical range: 1.5-4.5 t/m³).\n\n"
                                f"Do you want to continue anyway?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No
                            )
                            if reply == QMessageBox.StandardButton.No:
                                self.btn_run.setEnabled(True)
                                return
                        if max_den > 10:
                            reply = QMessageBox.question(
                                self, "Density Values Warning",
                                f"Density field '{density_field}' has values up to {max_den:.2f} t/m³.\n\n"
                                f"This seems unusually high (typical range: 1.5-4.5 t/m³).\n\n"
                                f"Do you want to continue anyway?",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes
                            )
                            if reply == QMessageBox.StandardButton.No:
                                self.btn_run.setEnabled(True)
                                return
                    except (ValueError, TypeError):
                        pass  # Skip validation if conversion fails

            # Validate volume field if using field mode
            if self.rb_volume_field.isChecked():
                volume_field = self.volume_field_combo.currentText()
                if volume_field and volume_field in self.block_model_data.columns:
                    vol_values = self.block_model_data[volume_field].dropna()

                    # Check if field exists and has data
                    if len(vol_values) == 0:
                        QMessageBox.warning(self, "Volume Field Empty",
                            f"Volume field '{volume_field}' has no valid data.\n\n"
                            "Please select a different volume field or use constant block dimensions.")
                        self.btn_run.setEnabled(True)
                        return

                    # Check if values are numeric
                    if not pd.api.types.is_numeric_dtype(vol_values):
                        QMessageBox.warning(self, "Volume Field Not Numeric",
                            f"Volume field '{volume_field}' does not contain numeric values.\n\n"
                            "Volume must be a numeric field (m³). Please select a valid volume field.")
                        self.btn_run.setEnabled(True)
                        return

                    # Check if values look like volume (positive, reasonable range)
                    try:
                        min_vol = float(vol_values.min())
                        max_vol = float(vol_values.max())
                        unique_count = int(vol_values.nunique())
                    except (ValueError, TypeError) as e:
                        QMessageBox.warning(self, "Volume Field Invalid",
                            f"Volume field '{volume_field}' contains non-numeric values.\n\n"
                            f"Error: {e}\n\n"
                            "Please select a valid numeric volume field.")
                        self.btn_run.setEnabled(True)
                        return

                    # Warning if values look suspicious (likely not volume)
                    if min_vol < 0:
                        reply = QMessageBox.question(
                            self, "Negative Volume Values",
                            f"Volume field '{volume_field}' contains negative values (min: {min_vol:.2f}).\n\n"
                            f"Volume should always be positive. This may not be a valid volume field.\n\n"
                            f"Do you want to continue anyway?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.No:
                            self.btn_run.setEnabled(True)
                            return

                    # Warning if all values are the same (might be a constant or placeholder)
                    if unique_count == 1 and len(vol_values) > 1:
                        logger.info(f"Volume field '{volume_field}' has constant value: {min_vol:.2f}")

                    # Warning if values look like classification codes (small integers with few unique values)
                    try:
                        looks_like_codes = (
                            max_vol <= 10 and
                            unique_count <= 10 and
                            np.allclose(vol_values, vol_values.astype(int))
                        )
                    except (ValueError, TypeError):
                        looks_like_codes = False

                    if looks_like_codes:
                        reply = QMessageBox.question(
                            self, "Volume Field Validation Warning",
                            f"Volume field '{volume_field}' has values ranging from {min_vol:.0f} to {max_vol:.0f} "
                            f"with only {unique_count} unique values.\n\n"
                            f"This looks like classification codes, not volume data!\n\n"
                            f"Expected volume values are typically larger (e.g., 500 m³ for a 10×10×5m block).\n\n"
                            f"Do you want to continue anyway?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.No
                        )
                        if reply == QMessageBox.StandardButton.No:
                            self.btn_run.setEnabled(True)
                            return
                else:
                    QMessageBox.warning(self, "Volume Field Missing",
                        f"Volume field '{volume_field}' not found in block model.\n\n"
                        "Please select a valid volume field or use constant block dimensions.")
                    self.btn_run.setEnabled(True)
                    return

            # Build configurations
            density_config = self._build_density_config()
            volume_config = self._build_volume_config()

            # Create engine
            engine = ResourceReportingEngine(
                block_model=self.block_model_data.copy(),
                class_field=self.class_combo.currentText(),
                grade_field=self.grade_combo.currentText(),
                domain_field=self.domain_combo.currentText() if self.domain_combo.currentText() != "(None)" else None,
                grade_is_pct=self.chk_grade_percent.isChecked(),
            )

            # Store for export
            self._current_engine = engine

            # Create worker and thread
            self._worker_thread = QThread()
            self._worker = ResourceReportingWorker(
                engine,
                density_config,
                volume_config
            )
            self._worker.moveToThread(self._worker_thread)

            # Connect signals
            self._worker_thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.error.connect(self._on_worker_error)
            self._worker.finished.connect(self._worker_thread.quit)
            self._worker.error.connect(self._worker_thread.quit)
            self._worker_thread.finished.connect(self._cleanup_worker)

            # Start
            logger.info("Starting resource reporting in background thread...")
            self._worker_thread.start()

        except Exception as e:
            logger.exception("Failed to start resource reporting")
            QMessageBox.critical(self, "Error", str(e))
            self.progress_lbl.setText("Error")
            self.btn_run.setEnabled(True)

    def _build_density_config(self):
        """Build density configuration from UI."""
        if self.rb_density_constant.isChecked():
            return DensityConfig(
                mode="constant",
                constant_value=self.density_spin.value()
            )
        elif self.rb_density_domain.isChecked():
            # Build domain table from UI table
            domain_data = []
            for row in range(self.domain_density_table.rowCount()):
                domain_item = self.domain_density_table.item(row, 0)
                density_item = self.domain_density_table.item(row, 1)

                if domain_item and density_item:
                    try:
                        domain = domain_item.text()
                        density = float(density_item.text())
                        # Use consistent column names: domain field name and DENSITY
                        domain_field_name = self.domain_combo.currentText()
                        domain_data.append({domain_field_name: domain, "DENSITY": density})
                    except ValueError:
                        raise ValueError(f"Invalid density value '{density_item.text()}' for domain '{domain}'")

            if not domain_data:
                raise ValueError("No domain density mappings defined. Please populate the domain table.")

            domain_table = pd.DataFrame(domain_data)
            return DensityConfig(
                mode="domain",
                domain_table=domain_table
            )
        elif self.rb_density_block.isChecked():
            return DensityConfig(
                mode="block",
                block_density_field=self.density_field_combo.currentText()
            )
        else:
            raise ValueError("No density mode selected")

    def _build_volume_config(self):
        """Build volume configuration from UI."""
        if self.rb_volume_field.isChecked():
            return VolumeConfig(
                mode="field",
                field_name=self.volume_field_combo.currentText()
            )
        elif self.rb_volume_constant.isChecked():
            return VolumeConfig(
                mode="constant",
                dx=self.dx_spin.value(),
                dy=self.dy_spin.value(),
                dz=self.dz_spin.value()
            )
        else:
            raise ValueError("No volume mode selected")

    def _on_worker_progress(self, pct: int, msg: str):
        """Handle progress updates."""
        self.progress_bar.setValue(pct)
        self.progress_lbl.setText(msg)

    def _on_worker_finished(self, result: ResourceSummaryResult):
        """Handle reporting completion."""
        self.summary_result = result
        self._update_table()

        # Store computed VOL, DEN, TONNES back into the block model
        self._store_computed_fields()

        self.reporting_complete.emit(result)
        self.btn_export.setEnabled(True)
        self.btn_export_excel.setEnabled(True)
        self.btn_generate_statement.setEnabled(True)
        self.btn_run.setEnabled(True)

        execution_time = result.metadata.get('execution_time_seconds', 0)
        self.progress_lbl.setText(f"Complete ({execution_time:.1f}s)")

        logger.info(f"Resource reporting complete: {len(result.rows)} classifications, {execution_time:.2f}s")

    def _on_worker_error(self, error_msg: str):
        """Handle reporting error."""
        QMessageBox.critical(self, "Resource Reporting Error", error_msg)
        self.progress_lbl.setText("Error")
        self.btn_run.setEnabled(True)

    def _store_computed_fields(self):
        """Store computed VOL, DEN, TONNES columns back into the block model."""
        try:
            if self._current_engine is None or self.block_model_data is None:
                return

            engine_df = self._current_engine.df

            # Only store if row counts match (engine was built from a copy)
            if len(engine_df) != len(self.block_model_data):
                logger.warning(
                    f"Cannot store computed fields: row count mismatch "
                    f"(engine={len(engine_df)}, block_model={len(self.block_model_data)})"
                )
                return

            stored_fields = []
            for col in ("VOL", "DEN", "TONNES"):
                if col in engine_df.columns:
                    self.block_model_data[col] = engine_df[col].values
                    stored_fields.append(col)

            if stored_fields:
                logger.info(f"Stored computed fields in block model: {stored_fields}")

                # Update the source registry entry so other panels can see new columns
                try:
                    source_name = self.source_combo.currentText()
                    if source_name and hasattr(self, '_block_model_sources'):
                        self._block_model_sources[source_name] = self.block_model_data
                except Exception:
                    pass  # Non-critical — fields are still on self.block_model_data

        except Exception as e:
            logger.warning(f"Could not store computed fields: {e}", exc_info=True)

    def _cleanup_worker(self):
        """Cleanup worker thread."""
        if hasattr(self, '_worker'):
            self._worker.deleteLater()
            self._worker = None
        if hasattr(self, '_worker_thread'):
            self._worker_thread.deleteLater()
            self._worker_thread = None

    def _update_table(self):
        """Update the results table with resource summary."""
        if not self.summary_result:
            return

        try:
            # Clear existing rows
            self.table.setRowCount(0)

            # Add data rows
            for row in self.summary_result.rows:
                row_idx = self.table.rowCount()
                self.table.insertRow(row_idx)

                self.table.setItem(row_idx, 0, QTableWidgetItem(row.classification))
                self.table.setItem(row_idx, 1, QTableWidgetItem(f"{row.n_blocks:,}"))
                self.table.setItem(row_idx, 2, QTableWidgetItem(f"{row.total_volume_m3:,.0f}"))
                self.table.setItem(row_idx, 3, QTableWidgetItem(f"{row.avg_density_t_per_m3:.2f}"))
                self.table.setItem(row_idx, 4, QTableWidgetItem(f"{row.total_tonnage_t:,.0f}"))
                self.table.setItem(row_idx, 5, QTableWidgetItem(f"{row.grade_pct:.2f}"))
                self.table.setItem(row_idx, 6, QTableWidgetItem(f"{row.contained_metal_t:,.0f}"))

            # Add totals rows
            if self.summary_result.totals_MI:
                self._add_totals_row(self.summary_result.totals_MI)

            if self.summary_result.totals_all:
                self._add_totals_row(self.summary_result.totals_all)

            # Resize columns to content
            self.table.resizeColumnsToContents()

            logger.info(f"Updated results table with {len(self.summary_result.rows)} classifications")

        except Exception as e:
            logger.error(f"Error updating results table: {e}", exc_info=True)
            QMessageBox.warning(self, "Update Error", f"Could not update results table: {e}")

    def _add_totals_row(self, totals_row: ResourceSummaryRow):
        """Add a totals row to the table."""
        row_idx = self.table.rowCount()
        self.table.insertRow(row_idx)

        # Create bold items for totals
        items = []
        for i, value in enumerate([
            totals_row.classification,
            f"{totals_row.n_blocks:,}",
            f"{totals_row.total_volume_m3:,.0f}",
            f"{totals_row.avg_density_t_per_m3:.2f}",
            f"{totals_row.total_tonnage_t:,.0f}",
            f"{totals_row.grade_pct:.2f}",
            f"{totals_row.contained_metal_t:,.0f}"
        ]):
            item = QTableWidgetItem(str(value))
            item.setFont(QFont("", -1, QFont.Weight.Bold))
            item.setBackground(QColor("#25252b"))
            items.append(item)
            self.table.setItem(row_idx, i, item)

    def _export_csv(self):
        """Export resource summary to CSV."""
        if not self.summary_result:
            QMessageBox.warning(self, "No Results", "Run resource summary first.")
            return

        # Get save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"resource_summary_{timestamp}.csv"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Resource Summary",
            default_name,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not path:
            return

        try:
            # Ensure .csv extension
            if not path.lower().endswith('.csv'):
                path += '.csv'

            # Prepare data for export
            rows_data = []
            for row in self.summary_result.rows:
                rows_data.append({
                    'Classification': row.classification,
                    'Blocks': row.n_blocks,
                    'Volume_m3': row.total_volume_m3,
                    'Avg_Density_t_per_m3': row.avg_density_t_per_m3,
                    'Tonnage_t': row.total_tonnage_t,
                    'Grade_pct': row.grade_pct,
                    'Contained_Metal_t': row.contained_metal_t,
                })

            # Add totals if available
            if self.summary_result.totals_MI:
                mi_totals = {
                    'Classification': self.summary_result.totals_MI.classification,
                    'Blocks': self.summary_result.totals_MI.n_blocks,
                    'Volume_m3': self.summary_result.totals_MI.total_volume_m3,
                    'Avg_Density_t_per_m3': self.summary_result.totals_MI.avg_density_t_per_m3,
                    'Tonnage_t': self.summary_result.totals_MI.total_tonnage_t,
                    'Grade_pct': self.summary_result.totals_MI.grade_pct,
                    'Contained_Metal_t': self.summary_result.totals_MI.contained_metal_t,
                }
                rows_data.append(mi_totals)

            if self.summary_result.totals_all:
                all_totals = {
                    'Classification': self.summary_result.totals_all.classification,
                    'Blocks': self.summary_result.totals_all.n_blocks,
                    'Volume_m3': self.summary_result.totals_all.total_volume_m3,
                    'Avg_Density_t_per_m3': self.summary_result.totals_all.avg_density_t_per_m3,
                    'Tonnage_t': self.summary_result.totals_all.total_tonnage_t,
                    'Grade_pct': self.summary_result.totals_all.grade_pct,
                    'Contained_Metal_t': self.summary_result.totals_all.contained_metal_t,
                }
                rows_data.append(all_totals)

            # Create DataFrame and export
            df = pd.DataFrame(rows_data)
            df.to_csv(path, index=False)

            # Export metadata to separate file
            metadata_path = path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(self.summary_result.metadata, f, indent=2, default=str)

            logger.info(f"Exported resource summary to {path} and metadata to {metadata_path}")

            QMessageBox.information(self, "Export Complete",
                f"Resource summary exported successfully!\n\n"
                f"Data: {path}\n"
                f"Metadata: {metadata_path}")

            self.progress_lbl.setText(f"Exported to {Path(path).name}")

        except Exception as e:
            logger.exception("Failed to export resource summary")
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _export_excel(self):
        """Export resource summary to Excel with formatting."""
        if not self.summary_result:
            QMessageBox.warning(self, "No Results", "Run resource summary first.")
            return

        # Get save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"resource_summary_{timestamp}.xlsx"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Resource Summary to Excel",
            default_name,
            "Excel Files (*.xlsx);;All Files (*)"
        )

        if not path:
            return

        try:
            # Ensure .xlsx extension
            if not path.lower().endswith('.xlsx'):
                path += '.xlsx'

            # Prepare data for export
            rows_data = []
            for row in self.summary_result.rows:
                rows_data.append({
                    'Classification': row.classification,
                    'Blocks': row.n_blocks,
                    'Volume (m³)': row.total_volume_m3,
                    'Avg Density (t/m³)': row.avg_density_t_per_m3,
                    'Tonnage (t)': row.total_tonnage_t,
                    'Grade (%)': row.grade_pct,
                    'Contained Metal (t)': row.contained_metal_t,
                })

            # Add totals if available
            if self.summary_result.totals_MI:
                mi_totals = {
                    'Classification': self.summary_result.totals_MI.classification,
                    'Blocks': self.summary_result.totals_MI.n_blocks,
                    'Volume (m³)': self.summary_result.totals_MI.total_volume_m3,
                    'Avg Density (t/m³)': self.summary_result.totals_MI.avg_density_t_per_m3,
                    'Tonnage (t)': self.summary_result.totals_MI.total_tonnage_t,
                    'Grade (%)': self.summary_result.totals_MI.grade_pct,
                    'Contained Metal (t)': self.summary_result.totals_MI.contained_metal_t,
                }
                rows_data.append(mi_totals)

            if self.summary_result.totals_all:
                all_totals = {
                    'Classification': self.summary_result.totals_all.classification,
                    'Blocks': self.summary_result.totals_all.n_blocks,
                    'Volume (m³)': self.summary_result.totals_all.total_volume_m3,
                    'Avg Density (t/m³)': self.summary_result.totals_all.avg_density_t_per_m3,
                    'Tonnage (t)': self.summary_result.totals_all.total_tonnage_t,
                    'Grade (%)': self.summary_result.totals_all.grade_pct,
                    'Contained Metal (t)': self.summary_result.totals_all.contained_metal_t,
                }
                rows_data.append(all_totals)

            # Create DataFrame
            df = pd.DataFrame(rows_data)

            # Export to Excel with formatting
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Resource Summary', index=False)

                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Resource Summary']

                # Format headers (bold, background color)
                from openpyxl.styles import Font, PatternFill, Alignment
                header_fill = PatternFill(start_color="4da6ff", end_color="4da6ff", fill_type="solid")
                header_font = Font(bold=True, color="FFFFFF")

                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal='center', vertical='center')

                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except Exception as e:
                            logger.error(f"Failed to calculate cell width: {e}", exc_info=True)
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

                # Add metadata sheet
                metadata_df = pd.DataFrame([
                    {'Parameter': k, 'Value': str(v)}
                    for k, v in self.summary_result.metadata.items()
                ])
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            logger.info(f"Exported resource summary to Excel: {path}")

            QMessageBox.information(self, "Export Complete",
                f"Resource summary exported successfully to:\n\n{path}")

            self.progress_lbl.setText(f"Exported to {Path(path).name}")

        except ImportError as e:
            logger.exception("Excel export requires openpyxl")
            QMessageBox.critical(self, "Export Error",
                "Excel export requires the 'openpyxl' library.\n\n"
                "Install it with: pip install openpyxl")
        except Exception as e:
            logger.exception("Failed to export resource summary to Excel")
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")

    def _format_tonnage(self, tonnes: float) -> str:
        """
        Format tonnage with appropriate units (t, kt, Mt, Gt).

        Follows mining industry conventions:
        - < 1,000 t: display as tonnes
        - 1,000 - 1,000,000 t: display as kt (kilotonnes)
        - 1,000,000 - 1,000,000,000 t: display as Mt (megatonnes)
        - >= 1,000,000,000 t: display as Gt (gigatonnes)
        """
        if tonnes < 1_000:
            return f"{tonnes:,.0f} t"
        elif tonnes < 1_000_000:
            return f"{tonnes / 1_000:,.2f} kt"
        elif tonnes < 1_000_000_000:
            return f"{tonnes / 1_000_000:,.2f} Mt"
        else:
            return f"{tonnes / 1_000_000_000:,.2f} Gt"

    def _generate_statement(self):
        """Generate human-readable resource summary statement."""
        if not self.summary_result:
            QMessageBox.warning(self, "No Results", "Run resource summary first.")
            return

        try:
            # Create summary text
            lines = []
            lines.append("RESOURCE SUMMARY STATEMENT")
            lines.append("=" * 50)
            lines.append("")

            # Get current settings
            grade_field = self.grade_combo.currentText()
            grade_unit = "%" if self.chk_grade_percent.isChecked() else ""

            # Add individual category statements
            for row in self.summary_result.rows:
                if row.n_blocks > 0:
                    tonnage_str = self._format_tonnage(row.total_tonnage_t)
                    metal_str = self._format_tonnage(row.contained_metal_t)
                    statement = (
                        f"The {row.classification} Resource is {tonnage_str} "
                        f"at {row.grade_pct:.2f}{grade_unit} {grade_field} "
                        f"containing {metal_str} of {grade_field}."
                    )
                    lines.append(statement)

            lines.append("")

            # Add totals if available
            if self.summary_result.totals_MI:
                mi = self.summary_result.totals_MI
                tonnage_str = self._format_tonnage(mi.total_tonnage_t)
                metal_str = self._format_tonnage(mi.contained_metal_t)
                statement = (
                    f"The Measured + Indicated Resource is {tonnage_str} "
                    f"at {mi.grade_pct:.2f}{grade_unit} {grade_field} "
                    f"containing {metal_str} of {grade_field}."
                )
                lines.append(statement)

            if self.summary_result.totals_all:
                total = self.summary_result.totals_all
                tonnage_str = self._format_tonnage(total.total_tonnage_t)
                metal_str = self._format_tonnage(total.contained_metal_t)
                statement = (
                    f"The Total Resource is {tonnage_str} "
                    f"at {total.grade_pct:.2f}{grade_unit} {grade_field} "
                    f"containing {metal_str} of {grade_field}."
                )
                lines.append(statement)

            # Add metadata
            lines.append("")
            lines.append("TECHNICAL DETAILS")
            lines.append("-" * 20)
            meta = self.summary_result.metadata
            lines.append(f"Classification Field: {self.class_combo.currentText()}")
            lines.append(f"Grade Field: {grade_field}")
            lines.append(f"Density Mode: {meta.get('density_mode', 'N/A')}")
            lines.append(f"Volume Mode: {meta.get('volume_mode', 'N/A')}")
            lines.append(f"Grade Units: {'Percent' if self.chk_grade_percent.isChecked() else 'Fractional'}")
            lines.append(f"Execution Time: {meta.get('execution_time_seconds', 0):.1f} seconds")
            lines.append(f"Numba Used: {meta.get('numba_used', False)}")
            lines.append(f"Generated: {meta.get('timestamp', 'N/A')}")

            # Create text dialog to display
            text_content = "\n".join(lines)

            # Create a dialog with a text area
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton, QFileDialog

            dialog = QDialog(self)
            dialog.setWindowTitle("Resource Summary Statement")
            dialog.setModal(True)
            dialog.resize(800, 600)

            layout = QVBoxLayout(dialog)

            # Text area
            text_edit = QTextEdit()
            text_edit.setPlainText(text_content)
            text_edit.setReadOnly(True)
            text_edit.setFontFamily("Courier New")  # Monospace for better formatting
            text_edit.setStyleSheet(f"""
                QTextEdit {{
                    background: {ModernColors.CARD_BG};
                    color: {ModernColors.TEXT_PRIMARY};
                    border: 1px solid {ModernColors.BORDER};
                    font-family: 'Courier New', monospace;
                    font-size: 10pt;
                }}
            """)
            layout.addWidget(text_edit)

            # Buttons
            button_layout = QHBoxLayout()

            # Copy to clipboard
            btn_copy = QPushButton("📋 Copy to Clipboard")
            btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(text_content))
            button_layout.addWidget(btn_copy)

            # Export to text file
            btn_export = QPushButton("💾 Export to File")
            btn_export.clicked.connect(lambda: self._export_statement_to_file(text_content))
            button_layout.addWidget(btn_export)

            button_layout.addStretch()

            # Close button
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.accept)
            button_layout.addWidget(btn_close)

            layout.addLayout(button_layout)

            dialog.exec()

        except Exception as e:
            logger.exception("Failed to generate statement")
            QMessageBox.critical(self, "Statement Error", f"Failed to generate statement:\n{e}")

    def _export_statement_to_file(self, content: str):
        """Export statement to a text file."""
        try:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Statement",
                "resource_statement.txt",
                "Text Files (*.txt);;All Files (*)"
            )

            if not path:
                return

            if not path.lower().endswith('.txt'):
                path += '.txt'

            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)

            QMessageBox.information(self, "Export Complete",
                f"Statement exported to:\n{path}")

        except Exception as e:
            logger.exception("Failed to export statement")
            QMessageBox.critical(self, "Export Error", f"Failed to export statement:\n{e}")
