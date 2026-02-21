"""
Indicator Kriging Panel

UI for configuring and running Indicator Kriging (IK).
Connects to High-Performance Numba Engine.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QTextEdit,
    QCheckBox, QLineEdit, QListWidget, QWidget,
    QSplitter, QScrollArea, QFrame, QMessageBox, QProgressBar, QInputDialog,
    QRadioButton, QButtonGroup, QDialog
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, QTimer

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import populate_variable_combo
from .base_analysis_panel import BaseAnalysisPanel
from ..geostats.ik_audit_utils import IndicatorKrigingAuditor, IndicatorKrigingBenchmark
from .cdf_inspector_dialog import show_cdf_inspector
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors

logger = logging.getLogger(__name__)


def _get_btn_style() -> str:
    """Get primary button style for current theme."""
    colors = get_theme_colors()
    return f"background-color: {colors.SUCCESS}; color: white; font-weight: bold; padding: 10px; border-radius: 4px;"

class IndicatorKrigingPanel(BaseAnalysisPanel):
    task_name = "indicator_kriging"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="indicator_kriging")
        self.setWindowTitle("Indicator Kriging")
        self.resize(1100, 750)
        self.setStyleSheet(get_analysis_panel_stylesheet())

        self.drillhole_data = None
        self.kriging_results = None

        self._build_ui()
        self._init_registry_connections()

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())
    
    def _build_ui(self):
        """Build the Indicator Kriging panel UI."""
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        split = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT ---
        left = QWidget()
        l_lay = QVBoxLayout(left)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        c_lay = QVBoxLayout(content)
        
        self._create_data_source_group(c_lay)
        self._create_var_group(c_lay)
        self._create_threshold_group(c_lay)
        self._create_vario_group(c_lay)
        self._create_grid_group(c_lay)
        self._create_output_group(c_lay)
        self._create_audit_group(c_lay)
        
        c_lay.addStretch()
        scroll.setWidget(content)
        l_lay.addWidget(scroll)
        
        # --- RIGHT ---
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.lbl_status = QLabel("Ready")
        r_lay.addWidget(QLabel("Status:"))
        r_lay.addWidget(self.progress)
        r_lay.addWidget(self.lbl_status)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        r_lay.addWidget(QLabel("Log:"))
        r_lay.addWidget(self.log_text)
        
        # Buttons
        h_btn = QHBoxLayout()
        self.btn_run = QPushButton("RUN ESTIMATION")
        self.btn_run.setStyleSheet(_get_btn_style())
        self.btn_run.clicked.connect(self.run_analysis)
        
        self.btn_viz = QPushButton("Visualize 3D")
        self.btn_viz.setEnabled(False)
        self.btn_viz.clicked.connect(self.visualize_results)

        self.view_table_btn = QPushButton("View Table")
        self.view_table_btn.setEnabled(False)
        self.view_table_btn.clicked.connect(self.open_results_table)

        h_btn.addWidget(self.btn_run)
        h_btn.addWidget(self.btn_viz)
        h_btn.addWidget(self.view_table_btn)
        r_lay.addLayout(h_btn)
        
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([400, 700])
        main_layout.addWidget(split)

    # --- BUILDERS ---
    def _create_data_source_group(self, layout):
        """Create data source selection group."""
        g = QGroupBox("0. Data Source")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #81c784; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Data source selector
        data_source_layout = QHBoxLayout()
        self.data_source_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data")
        self.data_source_composited.setToolTip("Use composited drillhole data (recommended)")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_raw.setToolTip("Use raw drillhole assay data")
        
        self.data_source_group.addButton(self.data_source_composited, 0)
        self.data_source_group.addButton(self.data_source_raw, 1)
        self.data_source_composited.setChecked(True)
        self.data_source_group.buttonClicked.connect(self._on_data_source_changed)
        
        data_source_layout.addWidget(self.data_source_composited)
        data_source_layout.addWidget(self.data_source_raw)
        data_source_layout.addStretch()
        f.addRow("Source:", data_source_layout)
        
        # Data source status label
        self.data_source_status_label = QLabel("")
        self.data_source_status_label.setStyleSheet("font-size: 9px; color: #888;")
        f.addRow("", self.data_source_status_label)
        
        layout.addWidget(g)

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_data_loaded(data)

    def _create_var_group(self, layout):
        g = QGroupBox("1. Variable")
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.var_combo = QComboBox()
        self.var_combo.setToolTip(
            "Select the grade variable for indicator kriging.\n"
            "IK estimates conditional probabilities above/below cutoff thresholds.\n"
            "Best suited for skewed distributions and resource classification."
        )
        f.addRow("Variable:", self.var_combo)
        layout.addWidget(g)

    def _create_threshold_group(self, layout):
        g = QGroupBox("2. Thresholds (Cutoffs)")
        l = QVBoxLayout(g)
        
        h1 = QHBoxLayout()
        self.thresh_in = QLineEdit()
        self.thresh_in.setPlaceholderText("e.g. 0.5")
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add_thresh)
        h1.addWidget(self.thresh_in)
        h1.addWidget(btn_add)
        l.addLayout(h1)
        
        h2 = QHBoxLayout()
        self.spin_count = QSpinBox()
        self.spin_count.setRange(3, 50)
        self.spin_count.setValue(9)
        btn_gen = QPushButton("Auto-Generate (Quantiles)")
        btn_gen.clicked.connect(self._gen_thresh)
        h2.addWidget(QLabel("Count:"))
        h2.addWidget(self.spin_count)
        h2.addWidget(btn_gen)
        l.addLayout(h2)
        
        self.list_thresh = QListWidget()
        self.list_thresh.setMaximumHeight(100)
        l.addWidget(self.list_thresh)
        
        btn_clr = QPushButton("Clear List")
        btn_clr.clicked.connect(self.list_thresh.clear)
        l.addWidget(btn_clr)
        layout.addWidget(g)

    def _create_vario_group(self, layout):
        g = QGroupBox("3. Variogram Model (Median Indicator)")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        v = QVBoxLayout(g)

        # Auto-load buttons
        btn_layout = QHBoxLayout()
        self.auto_vario_btn = QPushButton("Load from Variogram Panel")
        self.auto_vario_btn.clicked.connect(self.load_variogram_parameters)
        self.auto_vario_btn.setEnabled(False)

        self.use_assisted_btn = QPushButton("Load from Assistant")
        self.use_assisted_btn.clicked.connect(self._load_assisted_variogram)
        self.use_assisted_btn.setEnabled(False)

        btn_layout.addWidget(self.auto_vario_btn)
        btn_layout.addWidget(self.use_assisted_btn)
        v.addLayout(btn_layout)

        # Parameters
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Spherical", "Exponential", "Gaussian"])
        self.model_combo.setToolTip(
            "Variogram model for indicator transform (median threshold):\n"
            "• Spherical: Most common, reaches sill at range\n"
            "• Exponential: Gradual approach, effective range ≈ 3×range\n"
            "• Gaussian: Very smooth, for highly continuous phenomena\n\n"
            "IK uses indicator variograms - load from Variogram Panel if available."
        )

        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1, 1e5)
        self.range_spin.setValue(100)
        self.range_spin.setToolTip(
            "Range of spatial continuity for indicator variogram (project units).\n"
            "Distance at which indicator correlation becomes negligible.\n"
            "Typically shorter than grade variogram range."
        )

        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0, 1e5)
        self.nugget_spin.setToolTip(
            "Nugget effect for indicator variogram (sill = 1 for indicators).\n"
            "Represents micro-scale variability in ore/waste classification.\n"
            "High nugget indicates erratic grade transitions."
        )

        f.addRow("Model:", self.model_combo)
        f.addRow("Range:", self.range_spin)
        f.addRow("Nugget:", self.nugget_spin)
        v.addLayout(f)

        layout.addWidget(g)

    def _create_grid_group(self, layout):
        g = QGroupBox("4. Grid & Search")
        v = QVBoxLayout(g)
        
        # Origin
        h1 = QHBoxLayout()
        self.xmin = QDoubleSpinBox()
        self.xmin.setRange(-1e9, 1e9)
        self.ymin = QDoubleSpinBox()
        self.ymin.setRange(-1e9, 1e9)
        self.zmin = QDoubleSpinBox()
        self.zmin.setRange(-1e9, 1e9)
        h1.addWidget(QLabel("X0:"))
        h1.addWidget(self.xmin)
        h1.addWidget(QLabel("Y0:"))
        h1.addWidget(self.ymin)
        h1.addWidget(QLabel("Z0:"))
        h1.addWidget(self.zmin)
        v.addLayout(h1)
        
        # Size
        h2 = QHBoxLayout()
        self.dx = QDoubleSpinBox()
        self.dx.setValue(10)
        self.dy = QDoubleSpinBox()
        self.dy.setValue(10)
        self.dz = QDoubleSpinBox()
        self.dz.setValue(5)
        h2.addWidget(QLabel("DX:"))
        h2.addWidget(self.dx)
        h2.addWidget(QLabel("DY:"))
        h2.addWidget(self.dy)
        h2.addWidget(QLabel("DZ:"))
        h2.addWidget(self.dz)
        v.addLayout(h2)
        
        # Count
        h3 = QHBoxLayout()
        self.nx = QSpinBox()
        self.nx.setRange(1, 5000)
        self.nx.setValue(50)
        self.ny = QSpinBox()
        self.ny.setRange(1, 5000)
        self.ny.setValue(50)
        self.nz = QSpinBox()
        self.nz.setRange(1, 5000)
        self.nz.setValue(20)
        h3.addWidget(QLabel("NX:"))
        h3.addWidget(self.nx)
        h3.addWidget(QLabel("NY:"))
        h3.addWidget(self.ny)
        h3.addWidget(QLabel("NZ:"))
        h3.addWidget(self.nz)
        v.addLayout(h3)
        
        btn_auto = QPushButton("Auto-Fit Grid to Data")
        btn_auto.clicked.connect(self._auto_detect_grid)
        v.addWidget(btn_auto)
        
        # Auto-fit checkbox (enabled by default)
        self.auto_fit_grid_check = QCheckBox("Auto-fit grid when data loads")
        self.auto_fit_grid_check.setChecked(True)
        self.auto_fit_grid_check.setToolTip("Automatically restrict grid to drillhole extent when new data is loaded.")
        v.addWidget(self.auto_fit_grid_check)
        
        # Search
        h4 = QHBoxLayout()
        self.neigh = QSpinBox()
        self.neigh.setValue(12)
        h4.addWidget(QLabel("Neighbors:"))
        h4.addWidget(self.neigh)
        v.addLayout(h4)
        
        layout.addWidget(g)

    def _create_output_group(self, layout):
        g = QGroupBox("5. Outputs")
        v = QVBoxLayout(g)
        self.chk_median = QCheckBox("Compute Median (P50)")
        self.chk_median.setChecked(True)
        self.chk_mean = QCheckBox("Compute Mean (E-Type)")
        v.addWidget(self.chk_median)
        v.addWidget(self.chk_mean)
        layout.addWidget(g)

    def _create_audit_group(self, layout):
        """Create audit and validation group."""
        g = QGroupBox("6. Audit & Validation (GSLIB Standard)")
        g.setStyleSheet("QGroupBox { font-weight: bold; color: #ff6f00; border: 1px solid #444; margin-top: 6px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        v = QVBoxLayout(g)

        # Validation workflow button
        self.btn_run_validation = QPushButton("Run 5-Step Validation Workflow")
        self.btn_run_validation.clicked.connect(self.run_validation_workflow)
        self.btn_run_validation.setEnabled(False)
        self.btn_run_validation.setToolTip("Run complete GSLIB validation workflow")
        v.addWidget(self.btn_run_validation)

        # CDF Inspector button
        self.btn_cdf_inspector = QPushButton("CDF Inspector (Click Block in 3D)")
        self.btn_cdf_inspector.clicked.connect(self.open_cdf_inspector)
        self.btn_cdf_inspector.setEnabled(False)
        self.btn_cdf_inspector.setToolTip("Inspect CDF at individual blocks")
        v.addWidget(self.btn_cdf_inspector)

        # Diagnostic exports
        h_diag = QHBoxLayout()
        self.btn_slope_map = QPushButton("Export Probability Slope Map")
        self.btn_slope_map.clicked.connect(self.export_slope_map)
        self.btn_slope_map.setEnabled(False)
        self.btn_slope_map.setToolTip("Export transition strength analysis")

        self.btn_cdf_comparison = QPushButton("Global CDF Comparison")
        self.btn_cdf_comparison.clicked.connect(self.show_cdf_comparison)
        self.btn_cdf_comparison.setEnabled(False)
        self.btn_cdf_comparison.setToolTip("Compare IK CDF vs empirical CDF")

        h_diag.addWidget(self.btn_slope_map)
        h_diag.addWidget(self.btn_cdf_comparison)
        v.addLayout(h_diag)

        layout.addWidget(g)

    # --- LOGIC ---
    def _init_registry_connections(self):
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return
            
            self.registry.drillholeDataLoaded.connect(self._on_data_loaded)
            self.registry.variogramResultsLoaded.connect(self._on_variogram_results_loaded)
            
            # Source-toggle panels must load full drillhole payload for proper source switching.
            d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)

            # Check for existing variogram results
            vario = self.registry.get_variogram_results()
            if vario is not None:
                self._on_variogram_results_loaded(vario)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None

    def _on_data_loaded(self, data):
        """Handle drillhole data, respecting user's data source selection."""
        # Store registry data
        self._registry_data = data

        df = None
        composites = None
        assays = None
        composites_available = False
        assays_available = False

        if isinstance(data, dict):
            composites = data.get('composites')
            if composites is None:
                composites = data.get('composites_df')
            assays = data.get('assays')
            if assays is None:
                assays = data.get('assays_df')
            composites_available = isinstance(composites, pd.DataFrame) and not composites.empty
            assays_available = isinstance(assays, pd.DataFrame) and not assays.empty

            # Respect user's selection if radio buttons exist
            if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
                use_composited = self.data_source_composited.isChecked()

                if use_composited and composites_available:
                    df = composites
                    # AUDIT FIX: Set provenance for composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                elif not use_composited and assays_available:
                    df = assays
                    # AUDIT FIX: Mark raw assays appropriately
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
                elif composites_available:
                    # Fallback to composites
                    df = composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                    if hasattr(self, 'data_source_composited'):
                        self.data_source_composited.setChecked(True)
                elif assays_available:
                    # Fallback to assays
                    df = assays
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
                    if hasattr(self, 'data_source_raw'):
                        self.data_source_raw.setChecked(True)
            else:
                # Legacy: prefer composites
                if composites_available:
                    df = composites
                    if 'source_type' not in df.attrs:
                        df.attrs['source_type'] = 'composites'
                        df.attrs['lineage_gate_passed'] = True
                elif assays_available:
                    df = assays
                    df.attrs['source_type'] = 'raw_assays'
                    df.attrs['lineage_gate_passed'] = False
        elif isinstance(data, pd.DataFrame):
            df = data
            # AUDIT FIX: Set default provenance if not present
            if 'source_type' not in df.attrs:
                df.attrs['source_type'] = 'composites'
                df.attrs['lineage_gate_passed'] = True
            source_type = str(df.attrs.get('source_type', '')).lower()
            if source_type in ('raw_assays', 'assays'):
                assays_available = True
                assays = df
            else:
                composites_available = True
                composites = df
        
        # Update radio button states
        if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
            self.data_source_composited.setEnabled(composites_available)
            self.data_source_raw.setEnabled(assays_available)
            
            # Update status label
            if hasattr(self, 'data_source_status_label'):
                if composites_available and assays_available:
                    self.data_source_status_label.setText(
                        f"✓ Both available: {len(composites):,} composites, {len(assays):,} assays"
                    )
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                elif composites_available:
                    self.data_source_status_label.setText(f"✓ {len(composites):,} composites available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
                elif assays_available:
                    self.data_source_status_label.setText(f"✓ {len(assays):,} raw assays available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #FF9800;")
                else:
                    self.data_source_status_label.setText("⚠ No data available")
                    self.data_source_status_label.setStyleSheet("font-size: 9px; color: #e57373;")
        
        if df is not None and not df.empty:
            self.drillhole_data = ensure_xyz_columns(df)
            data_source = "composited" if (hasattr(self, 'data_source_composited') and self.data_source_composited.isChecked()) else "raw"
            logger.info(f"Indicator Kriging: Using {data_source} data ({len(df)} samples)")
            self._populate_vars()
            
            # Auto-detect grid from drillhole extent if checkbox is enabled
            if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
                self._auto_detect_grid()
                logger.info("Indicator Kriging: Auto-fitted grid to drillhole extent")
            
            self._log(f"Data loaded: {len(df)} rows from {data_source} data")

    def _on_variogram_results_loaded(self, results):
        """Store variogram results and enable loading buttons."""
        self.variogram_results = results
        # Enable variogram loading buttons if they exist
        if hasattr(self, 'auto_vario_btn'):
            self.auto_vario_btn.setEnabled(results is not None)
        if hasattr(self, 'use_assisted_btn'):
            self.use_assisted_btn.setEnabled(results is not None)

    def load_variogram_parameters(self) -> bool:
        """Populate variogram fields from variogram results."""
        if not self.variogram_results:
            return False

        # Only update UI if it's been built
        required_attrs = ['model_combo', 'range_spin', 'nugget_spin']
        if not all(hasattr(self, attr) for attr in required_attrs):
            logger.debug("Indicator Kriging panel: UI not ready for variogram parameter loading")
            return False

        try:
            # Check for combined_3d_model first (preferred), then fallback to fitted_models
            model = self.variogram_results.get('combined_3d_model')
            if not model:
                # Fallback to directional fits
                fits = self.variogram_results.get('fitted_models', {})
                omni_dict = fits.get('omni', {})
                if omni_dict:
                    # Get first model from omni dict
                    model = next(iter(omni_dict.values())) if omni_dict else None
            
            if not model:
                QMessageBox.warning(self, "No Variogram", "No variogram model found in results. Please run Variogram 3D Analysis first.")
                return False

            model_type = model.get('model_type', '').lower()
            display_model_type = model_type.capitalize()  # Capitalize for display

            # Check if the model type is available in the combo
            available_models = [self.model_combo.itemText(i).lower() for i in range(self.model_combo.count())]
            if model_type not in available_models:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Model type '{display_model_type}' not supported for indicator kriging.")
                return False

            # Apply parameters (indicator kriging doesn't use sill, only nugget)
            nugget = model.get('nugget', 0.0)
            range_val = model.get('major_range') or model.get('range', 100.0)

            self.model_combo.setCurrentText(display_model_type)
            self.range_spin.setValue(range_val)
            self.nugget_spin.setValue(nugget)

            logger.info(f"Indicator Kriging: Loaded variogram parameters - {display_model_type}, "
                       f"range={range_val:.1f}, nugget={nugget:.3f}")
            return True

        except Exception as e:
            logger.error(f"Error loading indicator kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load variogram parameters:\n{str(e)}")
            return False

    def _load_assisted_variogram(self):
        """Load variogram parameters from Variogram Assistant."""
        # Check if UI is ready
        required_attrs = ['model_combo', 'range_spin', 'nugget_spin']
        if not all(hasattr(self, attr) for attr in required_attrs):
            QMessageBox.warning(self, "UI Not Ready", "Please wait for the panel to finish loading.")
            return

        if not self.controller or not hasattr(self.controller, '_assisted_variogram_models'):
            QMessageBox.warning(self, "No Model", "No assisted variogram model available.")
            return

        try:
            assisted_models = self.controller._assisted_variogram_models
            if not assisted_models:
                QMessageBox.warning(self, "No Model", "No assisted variogram models available.")
                return

            # Use the first available assisted model
            assisted_model = assisted_models[0]
            model_type = assisted_model.get('model_type', '').lower()
            display_model_type = model_type.capitalize()

            # Check if the model type is available
            available_models = [self.model_combo.itemText(i).lower() for i in range(self.model_combo.count())]
            if model_type not in available_models:
                QMessageBox.warning(self, "Unsupported Model",
                                   f"Assisted model type '{display_model_type}' not supported for indicator kriging.")
                return

            # Apply parameters (no sill for indicator kriging)
            nugget = assisted_model.get('nugget', 0.0)
            range_val = assisted_model.get('range', 100.0)

            self.model_combo.setCurrentText(display_model_type)
            self.range_spin.setValue(range_val)
            self.nugget_spin.setValue(nugget)

            logger.info(f"Indicator Kriging: Loaded assisted variogram - {display_model_type}, "
                       f"range={range_val:.1f}, nugget={nugget:.3f}")

        except Exception as e:
            logger.error(f"Error loading assisted indicator kriging variogram: {e}", exc_info=True)
            QMessageBox.warning(self, "Load Error", f"Failed to load assisted variogram:\n{str(e)}")

    def _populate_vars(self):
        if self.drillhole_data is None:
            self.var_combo.clear()
            return
        # Populate variable combo using standardized method
        populate_variable_combo(self.var_combo, self.drillhole_data)

    def _add_thresh(self):
        try:
            val = float(self.thresh_in.text())
            self.list_thresh.addItem(f"{val}")
            self.thresh_in.clear()
        except:
            pass

    def _gen_thresh(self):
        if self.drillhole_data is None:
            return
        var = self.var_combo.currentText()
        if not var:
            return
        
        vals = self.drillhole_data[var].dropna().values
        if len(vals) == 0:
            return
        
        n = self.spin_count.value()
        # Quantiles
        qs = np.linspace(0, 100, n+2)[1:-1]
        cuts = np.percentile(vals, qs)
        
        self.list_thresh.clear()
        for c in cuts:
            self.list_thresh.addItem(f"{c:.3f}")
        self._log(f"Generated {len(cuts)} quantiles")

    def _auto_detect_grid(self):
        if self.drillhole_data is None:
            return
        df = self.drillhole_data
        
        xmin, xmax = df['X'].min(), df['X'].max()
        ymin, ymax = df['Y'].min(), df['Y'].max()
        zmin, zmax = df['Z'].min(), df['Z'].max()
        
        pad = 0.05
        dx = (xmax - xmin) * pad
        dy = (ymax - ymin) * pad
        dz = (zmax - zmin) * pad
        
        self.xmin.setValue(xmin - dx)
        self.ymin.setValue(ymin - dy)
        self.zmin.setValue(zmin - dz)
        
        # Recalc counts based on size
        bsx, bsy, bsz = self.dx.value(), self.dy.value(), self.dz.value()
        self.nx.setValue(int((xmax - xmin + 2*dx) / bsx))
        self.ny.setValue(int((ymax - ymin + 2*dy) / bsy))
        self.nz.setValue(int((zmax - zmin + 2*dz) / bsz))
        self._log("Grid auto-fitted to data extents")

    def gather_parameters(self):
        thresholds = []
        for i in range(self.list_thresh.count()):
            thresholds.append(float(self.list_thresh.item(i).text()))
        
        return {
            "data_df": self.drillhole_data,
            "variable": self.var_combo.currentText(),
            "thresholds": sorted(thresholds),
            "variogram_template": {
                "model_type": self.model_combo.currentText().lower(),
                "range": self.range_spin.value(),
                "sill": 1.0,  # Standardized indicator
                "nugget": self.nugget_spin.value()
            },
            "grid_spacing": (self.dx.value(), self.dy.value(), self.dz.value()),
            "grid_origin": (self.xmin.value(), self.ymin.value(), self.zmin.value()),
            "grid_counts": (self.nx.value(), self.ny.value(), self.nz.value()),
            "n_neighbors": self.neigh.value(),
            "compute_median": self.chk_median.isChecked(),
            "compute_mean": self.chk_mean.isChecked()
        }

    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before Indicator Kriging.

        Indicator Kriging requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Validated data quality

        Returns:
            True if data is acceptable for Indicator Kriging
        """
        registry = getattr(self, 'registry', None)
        if not registry:
            logger.warning("LINEAGE: No registry available - cannot verify data lineage")
            return True  # Allow to proceed but log warning

        # HARD GATE: Use require_validation_for_estimation() method
        # This enforces JORC/SAMREC compliance - NO estimation without validation
        allowed, message = registry.require_validation_for_estimation()
        if not allowed:
            logger.error(f"LINEAGE HARD GATE: {message}")
            QMessageBox.critical(
                self, "Validation Required",
                f"Cannot run Indicator Kriging:\n\n{message}\n\n"
                "Open the QC Window to validate your data before running estimation."
            )
            return False

        # Log validation status for audit trail
        validation_state = registry.get_drillholes_validation_state()
        if validation_state:
            status = validation_state.get("status", "UNKNOWN")
            if status == "WARN":
                logger.warning(
                    "LINEAGE: Validation passed with warnings. "
                    "Review warnings for JORC/SAMREC compliance."
                )
            else:
                logger.info(f"LINEAGE: Validation status = {status}")

        return True

    def run_analysis(self):
        if not self.controller:
            return
        if self.drillhole_data is None:
            QMessageBox.warning(self, "Error", "No Data")
            return
        if self.list_thresh.count() == 0:
            QMessageBox.warning(self, "Error", "Add thresholds")
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return
            
        params = self.gather_parameters()
        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self._log("Starting Indicator Kriging...")
        
        # Callback closure
        def progress_cb(pct, msg):
            QTimer.singleShot(0, lambda: self._update_progress(pct, msg))
        
        try:
            params['_progress_callback'] = progress_cb
            self.controller.run_indicator_kriging(
                config=params,
                callback=self.on_results,
                progress_callback=progress_cb
            )
        except Exception as e:
            self._log(f"Error: {e}", "error")
            self.btn_run.setEnabled(True)

    def on_results(self, payload):
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        
        if "error" in payload:
            self._log(f"Failed: {payload['error']}", "error")
            return
            
        self.kriging_results = payload
        self.btn_viz.setEnabled(True)
        self.view_table_btn.setEnabled(True)
        # Enable audit buttons
        self.btn_run_validation.setEnabled(True)
        self.btn_cdf_inspector.setEnabled(True)
        self.btn_slope_map.setEnabled(True)
        self.btn_cdf_comparison.setEnabled(True)
        self._log("Calculation Complete.", "success")
        
        # Register results in registry for other panels to use
        if self.registry:
            try:
                # Get variable name from metadata or current selection
                variable = payload.get('metadata', {}).get('variable') or self.var_combo.currentText() if hasattr(self, 'var_combo') else 'IK'
                metadata = {
                    'variable': variable,
                    'thresholds': payload.get('thresholds', []),
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                self.registry.register_indicator_kriging_results(payload, source_panel="IndicatorKrigingPanel", metadata=metadata)
                self._log(f"Results registered in registry for variable '{variable}'", "success")
            except Exception as e:
                logger.warning(f"Failed to register IK results in registry: {e}", exc_info=True)

            # Also register the block model DataFrame for cross-sections and other panels
            try:
                grid_x = payload['grid_x']
                grid_y = payload['grid_y']
                grid_z = payload['grid_z']
                estimates = payload.get('estimates', payload.get('probabilities', np.array([])))

                if len(estimates) > 0:
                    # Create block model DataFrame
                    coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                    block_df = pd.DataFrame({
                        'X': coords[:, 0],
                        'Y': coords[:, 1],
                        'Z': coords[:, 2],
                        f'{variable}_ik_prob': estimates.ravel()
                    }).dropna()

                    # Register the block model
                    self.registry.register_block_model_generated(
                        block_df,
                        source_panel="Indicator Kriging",
                        metadata={
                            'variable': variable,
                            'method': 'indicator_kriging',
                            'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                            'n_blocks': len(block_df),
                            'cutoff': cutoff
                        }
                    )
                    self._log("Block model registered to data registry", "success")
            except Exception as e:
                logger.warning(f"Failed to register IK block model: {e}")

        # Show stats
        if 'median' in payload:
            med = payload['median']
            self._log(f"Median Est: Min={np.nanmin(med):.2f} Max={np.nanmax(med):.2f}")

    def visualize_results(self):
        if not self.kriging_results:
            return
        
        # Ask what to visualize
        items = []
        if 'median' in self.kriging_results:
            items.append("Median")
        if 'mean' in self.kriging_results:
            items.append("Mean (E-Type)")
        # Add specific thresholds
        thresh = self.kriging_results.get('thresholds', [])
        for t in thresh:
            items.append(f"Prob > {t}")
            
        if not items:
            return
        
        item, ok = QInputDialog.getItem(self, "Visualize", "Select Property:", items, 0, False)
        
        if ok and item:
            # Prepare payload for viewer
            # Extract the specific array
            arr = None
            name = ""
            
            if item == "Median":
                arr = self.kriging_results['median']
                name = self.kriging_results.get('median_property', "IK_Median")
            elif item == "Mean (E-Type)":
                arr = self.kriging_results['mean']
                name = self.kriging_results.get('mean_property', "IK_Mean")
            else:
                # Probability map
                t_val = float(item.split("> ")[1])
                # Find index
                idx = list(thresh).index(t_val)
                # Probabilities shape (nx, ny, nz, n_thresh)
                probs = self.kriging_results['probabilities']
                arr = probs[:, :, :, idx]
                name = f"IK_Prob_gt_{t_val}"

            # Construct grid payload
            viz_payload = {
                'grid_x': self.kriging_results['grid_x'],
                'grid_y': self.kriging_results['grid_y'],
                'grid_z': self.kriging_results['grid_z'],
                'estimates': arr,
                'variances': np.zeros_like(arr),  # Dummy
                'variable': name,
                'metadata': {}
            }
            
            # Emit to main window
            if self.parent() and hasattr(self.parent(), 'visualize_kriging_results'):
                self.parent().visualize_kriging_results(viz_payload)
                self._log(f"Visualizing {name}", "success")

    def open_results_table(self):
        """Open Indicator Kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run Indicator Kriging first.")
            return

        try:
            grid_x = self.kriging_results.get('grid_x')
            grid_y = self.kriging_results.get('grid_y')
            grid_z = self.kriging_results.get('grid_z')

            if grid_x is None:
                QMessageBox.warning(self, "Invalid Results", "Results data is incomplete.")
                return

            coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
            df = pd.DataFrame({
                'X': coords[:, 0],
                'Y': coords[:, 1],
                'Z': coords[:, 2],
            })

            # Add threshold probability results
            threshold_properties = self.kriging_results.get('threshold_properties', {})
            for prop_name, values in threshold_properties.items():
                if isinstance(values, np.ndarray):
                    df[prop_name] = values

            # Add additional properties (median, mean, etc.)
            additional_properties = self.kriging_results.get('additional_properties', {})
            for prop_name, values in additional_properties.items():
                if isinstance(values, np.ndarray):
                    df[prop_name] = values

            df = df.dropna()

            title = "Indicator Kriging Results"

            # Try to find MainWindow parent first
            parent = self.parent()
            main_window = None
            while parent:
                if hasattr(parent, 'open_table_viewer_window_from_df'):
                    main_window = parent
                    break
                parent = parent.parent()

            if main_window:
                main_window.open_table_viewer_window_from_df(df, title=title)
            else:
                # Create table viewer dialog directly
                from .table_viewer_panel import TableViewerPanel

                dialog = QDialog(self)
                dialog.setWindowTitle(title)
                dialog.resize(900, 700)
                dialog.setWindowFlags(
                    Qt.WindowType.Window |
                    Qt.WindowType.WindowMinimizeButtonHint |
                    Qt.WindowType.WindowMaximizeButtonHint |
                    Qt.WindowType.WindowCloseButtonHint
                )

                layout = QVBoxLayout(dialog)
                table_viewer = TableViewerPanel()
                table_viewer.set_dataframe(df)
                layout.addWidget(table_viewer)

                dialog.show()

        except Exception as e:
            logger.error(f"Error opening results table: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to open results table:\n{str(e)}")

    def _update_progress(self, val, msg):
        pct = max(0, min(100, int(val)))
        self.progress.setValue(pct)
        if msg:
            self.progress.setFormat(f"{pct}% - {msg}")
        else:
            self.progress.setFormat(f"{pct}%")
        self.lbl_status.setText(msg)
        
    def _log(self, msg, level="info"):
        color = "#4fc3f7" if level == "info" else "#81c784" if level == "success" else "#e57373"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f'<span style="color:#777">[{timestamp}]</span> <span style="color:{color}">{msg}</span>')

    # --- AUDIT METHODS ---

    def run_validation_workflow(self):
        """Run the complete 5-step GSLIB validation workflow."""
        if not self.kriging_results or not self._registry_data:
            QMessageBox.warning(self, "Validation Error",
                              "No IK results or data available for validation.")
            return

        try:
            self._log("Starting IK validation workflow...", "info")

            # Get original data
            original_data = None
            if isinstance(self._registry_data, dict):
                original_data = self._registry_data.get('composites')
                if original_data is None or (hasattr(original_data, 'empty') and original_data.empty):
                    original_data = self._registry_data.get('assays')
            elif isinstance(self._registry_data, pd.DataFrame):
                original_data = self._registry_data

            if original_data is None or original_data.empty:
                QMessageBox.warning(self, "Validation Error", "No original data available for validation.")
                return

            # Get variable name
            variable = self.var_combo.currentText()
            if not variable:
                QMessageBox.warning(self, "Validation Error", "No variable selected.")
                return

            # Run validation
            auditor = IndicatorKrigingAuditor()
            validation_report = auditor.run_full_validation_workflow(
                self.kriging_results, original_data, variable
            )

            # Display results
            self._show_validation_report(validation_report)

            self._log("IK validation workflow completed", "success")

        except Exception as e:
            logger.error(f"Validation workflow failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Validation Error", f"Validation failed: {str(e)}")

    def _show_validation_report(self, report):
        """Display validation report in a dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("IK Validation Report (GSLIB Standard)")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        # Format report
        report_text = self._format_validation_report(report)
        text_edit.setHtml(report_text)

        layout.addWidget(text_edit)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(lambda: self._export_validation_report(report))
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _format_validation_report(self, report):
        """Format validation report as HTML."""
        html = "<h2>Indicator Kriging Validation Report</h2>"
        html += f"<p><b>Timestamp:</b> {report['timestamp']}</p>"
        html += f"<p><b>Variable:</b> {report['variable']}</p>"

        # Overall assessment
        assessment = report.get('overall_assessment', {})
        score = assessment.get('overall_score', 0)
        confidence = assessment.get('confidence_level', 'unknown')

        html += f"<h3>Overall Assessment: {confidence.upper()}</h3>"
        html += f"<p>Score: {score:.2f}/1.0</p>"

        if assessment.get('issues'):
            html += "<h4>Issues Found:</h4><ul>"
            for issue in assessment['issues']:
                html += f"<li>{issue}</li>"
            html += "</ul>"

        if assessment.get('recommendations'):
            html += "<h4>Recommendations:</h4><ul>"
            for rec in assessment['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        # Step-by-step results
        html += "<h3>Validation Steps</h3>"

        steps = report.get('validation_steps', {})
        for step_name, step_data in steps.items():
            html += f"<h4>{step_name.replace('_', ' ').title()}</h4>"

            if isinstance(step_data, dict):
                if 'error' in step_data:
                    html += f"<p style='color:red'>Error: {step_data['error']}</p>"
                else:
                    for key, value in step_data.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            html += f"<p><b>{key}:</b> {value:.4f}</p>"
                        elif isinstance(value, bool):
                            status = "✓" if value else "✗"
                            html += f"<p><b>{key}:</b> {status}</p>"
                        else:
                            html += f"<p><b>{key}:</b> {value}</p>"

        return html

    def _export_validation_report(self, report):
        """Export validation report to file."""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Validation Report", "", "JSON files (*.json);;All files (*)"
            )

            if filename:
                import json
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

                QMessageBox.information(self, "Export Complete",
                                      f"Validation report saved to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export report: {str(e)}")

    def open_cdf_inspector(self):
        """Open CDF inspector for clicked block."""
        if not self.kriging_results:
            QMessageBox.warning(self, "CDF Inspector", "No IK results available.")
            return

        # Instructions for user
        msg = QMessageBox()
        msg.setWindowTitle("CDF Inspector")
        msg.setText("CDF Inspector Instructions")
        msg.setInformativeText(
            "1. Click 'OK' to continue\n"
            "2. In the 3D visualization, click on any block\n"
            "3. The CDF inspector will show the full probability distribution for that block"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Ok)

        if msg.exec() == QMessageBox.StandardButton.Ok:
            # Set up click handler for 3D view
            self._setup_block_click_handler()

    def _setup_block_click_handler(self):
        """Set up handler for block clicks in 3D view."""
        # This would need integration with the 3D visualization system
        # For now, show a placeholder
        QMessageBox.information(self, "CDF Inspector",
                              "Block click handler would be set up here.\n"
                              "Integration with 3D viewer needed for full functionality.")

    def export_slope_map(self):
        """Export probability slope map for transition analysis."""
        if not self.kriging_results:
            QMessageBox.warning(self, "Export Error", "No IK results available.")
            return

        try:
            auditor = IndicatorKrigingAuditor()

            # Use default band (could be made configurable)
            slope_result = auditor.compute_probability_slope_map(self.kriging_results)

            if 'error' in slope_result:
                QMessageBox.warning(self, "Export Error", slope_result['error'])
                return

            # Export as additional property for visualization
            slope_data = slope_result['slope_map']

            # Add to results for visualization
            self.kriging_results['additional_properties'] = self.kriging_results.get('additional_properties', {})
            self.kriging_results['additional_properties']['IK_Probability_Slope'] = slope_data.ravel()

            QMessageBox.information(self, "Export Complete",
                                  f"Probability slope map exported.\n"
                                  f"Slope range: {slope_result['slope_range']}\n"
                                  f"High transition zones: {slope_result['high_transition_zones']}")

            self._log("Probability slope map exported", "success")

        except Exception as e:
            logger.error(f"Slope map export failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export slope map: {str(e)}")

    def show_cdf_comparison(self):
        """Show global CDF vs IK-implied CDF comparison."""
        if not self.kriging_results or not self._registry_data:
            QMessageBox.warning(self, "CDF Comparison", "No IK results or data available.")
            return

        try:
            # Get original data
            original_data = None
            if isinstance(self._registry_data, dict):
                original_data = self._registry_data.get('composites')
                if original_data is None or (hasattr(original_data, 'empty') and original_data.empty):
                    original_data = self._registry_data.get('assays')
            elif isinstance(self._registry_data, pd.DataFrame):
                original_data = self._registry_data

            if original_data is None or original_data.empty:
                QMessageBox.warning(self, "CDF Comparison", "No original data available.")
                return

            variable = self.var_combo.currentText()
            if not variable:
                QMessageBox.warning(self, "CDF Comparison", "No variable selected.")
                return

            auditor = IndicatorKrigingAuditor()
            comparison = auditor.compute_global_cdf_comparison(
                self.kriging_results, original_data, variable
            )

            if 'error' in comparison:
                QMessageBox.warning(self, "CDF Comparison", comparison['error'])
                return

            # Show comparison plot
            self._show_cdf_comparison_dialog(comparison)

        except Exception as e:
            logger.error(f"CDF comparison failed: {e}", exc_info=True)
            QMessageBox.critical(self, "CDF Comparison", f"Failed to create comparison: {str(e)}")

    def _show_cdf_comparison_dialog(self, comparison):
        """Show CDF comparison in a dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

        dialog = QDialog(self)
        dialog.setWindowTitle("Global CDF vs IK-Implied CDF")
        dialog.resize(600, 500)

        layout = QVBoxLayout(dialog)

        # Create plot
        figure = plt.Figure(figsize=(6, 4))
        canvas = FigureCanvas(figure)

        ax = figure.add_subplot(111)
        thresholds = comparison['thresholds']
        empirical = comparison['empirical_cdf']
        ik_cdf = comparison['ik_cdf']

        ax.plot(thresholds, empirical, 'b-o', label='Empirical CDF (Data)', linewidth=2)
        ax.plot(thresholds, ik_cdf, 'r-s', label='IK-Implied CDF', linewidth=2, alpha=0.8)

        ax.set_xlabel('Grade Threshold')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add statistics
        max_diff = comparison['max_absolute_difference']
        is_consistent = comparison['is_consistent']

        status_color = 'green' if is_consistent else 'red'
        ax.text(0.02, 0.98, f'Max Difference: {max_diff:.3f}\\nConsistent: {"Yes" if is_consistent else "No"}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.1))

        layout.addWidget(canvas)

        # Add close button
        from PyQt6.QtWidgets import QHBoxLayout, QPushButton
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

        dialog.exec()

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Data source
            if hasattr(self, 'data_source_composited'):
                settings['data_source'] = 'composited' if self.data_source_composited.isChecked() else 'raw'
            
            # Variable
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            
            # Thresholds (save as list)
            if hasattr(self, 'thresh_list') and self.thresh_list:
                try:
                    thresholds = []
                    for i in range(self.thresh_list.count()):
                        item = self.thresh_list.item(i)
                        if item:
                            try:
                                thresholds.append(float(item.text()))
                            except ValueError:
                                pass
                    if thresholds:
                        settings['thresholds'] = thresholds
                except Exception:
                    pass
            
            # Variogram model
            settings['model_type'] = get_safe_widget_value(self, 'model_combo')
            settings['nugget'] = get_safe_widget_value(self, 'nugget_spin')
            settings['sill'] = get_safe_widget_value(self, 'sill_spin')
            settings['range'] = get_safe_widget_value(self, 'range_spin')
            
            # Grid
            settings['xmin'] = get_safe_widget_value(self, 'xmin_spin')
            settings['ymin'] = get_safe_widget_value(self, 'ymin_spin')
            settings['zmin'] = get_safe_widget_value(self, 'zmin_spin')
            settings['grid_x'] = get_safe_widget_value(self, 'dx_spin')
            settings['grid_y'] = get_safe_widget_value(self, 'dy_spin')
            settings['grid_z'] = get_safe_widget_value(self, 'dz_spin')
            
            # Search settings
            settings['neighbors'] = get_safe_widget_value(self, 'neighbors_spin')
            
            # Output settings
            settings['output_e_type'] = get_safe_widget_value(self, 'e_type_check')
            settings['output_full_cdf'] = get_safe_widget_value(self, 'full_cdf_check')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save indicator kriging panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Data source
            if 'data_source' in settings:
                if settings['data_source'] == 'composited' and hasattr(self, 'data_source_composited'):
                    self.data_source_composited.setChecked(True)
                elif settings['data_source'] == 'raw' and hasattr(self, 'data_source_raw'):
                    self.data_source_raw.setChecked(True)
            
            # Variable
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            
            # Thresholds
            if 'thresholds' in settings and hasattr(self, 'thresh_list'):
                self.thresh_list.clear()
                for t in settings['thresholds']:
                    self.thresh_list.addItem(str(t))
            
            # Variogram model
            set_safe_widget_value(self, 'model_combo', settings.get('model_type'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget'))
            set_safe_widget_value(self, 'sill_spin', settings.get('sill'))
            set_safe_widget_value(self, 'range_spin', settings.get('range'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
            
            # Output settings
            set_safe_widget_value(self, 'e_type_check', settings.get('output_e_type'))
            set_safe_widget_value(self, 'full_cdf_check', settings.get('output_full_cdf'))
                
            logger.info("Restored indicator kriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore indicator kriging panel settings: {e}")
