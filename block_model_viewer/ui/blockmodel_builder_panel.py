"""
Block Model Builder Panel

UI for building regular 3D block models.
Refactored for Modern UX/UI.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QDoubleSpinBox, QPushButton, QMessageBox, QDialog,
    QTextEdit, QFileDialog, QSpinBox, QSplitter, QScrollArea, QFrame, QFormLayout,
    QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from block_model_viewer.models import blockmodel_builder as bmb
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)

# Evaluation method options
EVALUATION_METHODS = [
    "-- Select Method --",
    "Ordinary Kriging",
    "Simple Kriging",
    "Universal Kriging",
    "Co-Kriging",
    "Indicator Kriging",
    "RBF Interpolation",
    "SGSIM (Simulation)",
]


# Worker logic moved to DataController._prepare_build_block_model_payload
# This ensures pure computation with no access to DataRegistry or Qt objects


class BlockModelBuilderPanel(QDialog):
    """Main panel for Block Model Builder."""
    request_visualization = pyqtSignal(object, str, str)  # grid, property_name, method_name
    
    def __init__(self, parent=None, controller=None):
        super().__init__(parent)
        self.setWindowTitle("Block Model Builder")
        self.resize(900, 600)

        # Controller (injected from MainWindow)
        self.controller = controller
        
        self.estimation_data = None
        self.drillhole_data = None
        self.drillhole_extents = None  # {'xmin': ..., 'xmax': ..., 'ymin': ..., 'ymax': ..., 'zmin': ..., 'zmax': ...}
        self.block_model_df = None
        self.block_grid = None
        self.block_info = None
        self.grade_col = None
        self.var_col = None
        self.current_method = None  # Store the current method name for layer naming
        
        # Store results from different estimation methods
        self._kriging_results = {}  # method_name -> results dict
        self._available_properties = []  # List of (method, property_name, var_name, data_key) tuples
        
        self._setup_ui()
        self._init_registry()

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.krigingResultsLoaded.connect(self._on_krig_loaded)
                self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                if hasattr(self.registry, 'drillholeDataLoaded'):
                    self.registry.drillholeDataLoaded.connect(self._on_drillhole_loaded)
                
                # Connect additional result signals if available
                if hasattr(self.registry, 'simpleKrigingResultsLoaded'):
                    self.registry.simpleKrigingResultsLoaded.connect(self._on_simple_kriging_loaded)
                if hasattr(self.registry, 'universalKrigingResultsLoaded'):
                    self.registry.universalKrigingResultsLoaded.connect(self._on_universal_kriging_loaded)
                if hasattr(self.registry, 'cokrigingResultsLoaded'):
                    self.registry.cokrigingResultsLoaded.connect(self._on_cokriging_loaded)
                if hasattr(self.registry, 'indicatorKrigingResultsLoaded'):
                    self.registry.indicatorKrigingResultsLoaded.connect(self._on_indicator_kriging_loaded)
                if hasattr(self.registry, 'rbfResultsLoaded'):
                    self.registry.rbfResultsLoaded.connect(self._on_rbf_loaded)

                # Refresh all available data sources
                self._refresh_available_data()
                
                # Load drillhole data for extent detection
                dh = self.registry.get_drillhole_data()
                if dh:
                    self._on_drillhole_loaded(dh)
        except Exception as e:
            logger.debug(f"Registry init failed: {e}")

    def get_registry(self):
        """Resolve DataRegistry from controller or parent MainWindow."""
        if self.controller and hasattr(self.controller, "registry"):
            return self.controller.registry
        parent = self.parent()
        while parent:
            if hasattr(parent, "controller") and getattr(parent, "controller"):
                ctrl = parent.controller
                if hasattr(ctrl, "registry"):
                    return ctrl.registry
            if hasattr(parent, "_registry") and getattr(parent, "_registry", None):
                return parent._registry
            parent = parent.parent()
        return None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT (Config)
        left = QWidget()
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)
        
        self._create_data_source_group(s_lay)
        self._create_dims_group(s_lay)
        self._create_volume_tonnage_group(s_lay)
        self._create_limits_group(s_lay)
        
        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # RIGHT (Preview & Actions)
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        prev_grp = QGroupBox("Model Preview")
        p_lay = QVBoxLayout(prev_grp)
        self.preview_lbl = QLabel("Load data to see preview.")
        self.preview_lbl.setWordWrap(True)
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        p_lay.addWidget(self.preview_lbl)
        r_lay.addWidget(prev_grp)
        
        log_grp = QGroupBox("Build Log")
        l_lay_log = QVBoxLayout(log_grp)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet(f"background-color: {ModernColors.ELEVATED_BG}; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas;")
        l_lay_log.addWidget(self.log)
        r_lay.addWidget(log_grp)
        
        act_lay = QHBoxLayout()
        self.build_btn = QPushButton("BUILD MODEL")
        self.build_btn.setStyleSheet(f"background-color: {ModernColors.SUCCESS}; color: white; font-weight: bold; padding: 12px;")
        self.build_btn.clicked.connect(self.build_block_model)
        self.build_btn.setEnabled(False)
        
        self.viz_btn = QPushButton("Visualize")
        self.viz_btn.clicked.connect(self.visualize_block_model)
        self.viz_btn.setEnabled(False)
        self.exp_btn = QPushButton("Export")
        self.exp_btn.clicked.connect(self.export_block_model)
        self.exp_btn.setEnabled(False)
        
        act_lay.addWidget(self.build_btn, stretch=2)
        act_lay.addWidget(self.viz_btn)
        act_lay.addWidget(self.exp_btn)
        r_lay.addLayout(act_lay)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    def _create_data_source_group(self, layout):
        """Create data source selection group with evaluation method and property selectors."""
        g = QGroupBox("0. Data Source (Evaluation Method)")
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.SUCCESS}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        v_lay = QVBoxLayout(g)
        
        # Evaluation method selector
        method_lay = QHBoxLayout()
        method_lay.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(EVALUATION_METHODS)
        self.method_combo.setToolTip(
            "Select the estimation/simulation method to use for building the block model.\n"
            "Each method produces different results - ensure you run the appropriate analysis first."
        )
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_lay.addWidget(self.method_combo, stretch=1)
        v_lay.addLayout(method_lay)
        
        # Property selector (populated based on selected method)
        prop_lay = QHBoxLayout()
        prop_lay.addWidget(QLabel("Property:"))
        self.property_combo = QComboBox()
        self.property_combo.setToolTip(
            "Select the estimation property to use for block values.\n"
            "Both Estimate and Variance will be added as separate properties\n"
            "in the block model for visualization and analysis."
        )
        self.property_combo.currentIndexChanged.connect(self._on_property_changed)
        prop_lay.addWidget(self.property_combo, stretch=1)
        v_lay.addLayout(prop_lay)
        
        # Load/Refresh button
        btn_lay = QHBoxLayout()
        self.load_data_btn = QPushButton("Load Results from Method")
        self.load_data_btn.setStyleSheet(f"background-color: {ModernColors.INFO}; color: white; padding: 6px;")
        self.load_data_btn.setToolTip("Load estimation results from the selected method into the builder.")
        self.load_data_btn.clicked.connect(self._load_data_from_method)
        btn_lay.addWidget(self.load_data_btn)
        
        self.refresh_data_btn = QPushButton("Refresh")
        self.refresh_data_btn.setToolTip("Refresh available methods and properties from data registry.")
        self.refresh_data_btn.clicked.connect(self._refresh_available_data)
        btn_lay.addWidget(self.refresh_data_btn)
        v_lay.addLayout(btn_lay)
        
        # Status label
        self.data_status_lbl = QLabel("No estimation data loaded")
        self.data_status_lbl.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        self.data_status_lbl.setWordWrap(True)
        v_lay.addWidget(self.data_status_lbl)
        
        layout.addWidget(g)

    def _create_dims_group(self, layout):
        g = QGroupBox("1. Block Dimensions")
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.INFO}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        h = QHBoxLayout()
        self.dx = QDoubleSpinBox()
        self.dx.setRange(0.1, 1000)
        self.dx.setValue(25)
        self.dx.valueChanged.connect(self._update_preview)
        self.dy = QDoubleSpinBox()
        self.dy.setRange(0.1, 1000)
        self.dy.setValue(25)
        self.dy.valueChanged.connect(self._update_preview)
        self.dz = QDoubleSpinBox()
        self.dz.setRange(0.1, 1000)
        self.dz.setValue(10)
        self.dz.valueChanged.connect(self._update_preview)
        
        h.addWidget(QLabel("DX:"))
        h.addWidget(self.dx)
        h.addWidget(QLabel("DY:"))
        h.addWidget(self.dy)
        h.addWidget(QLabel("DZ:"))
        h.addWidget(self.dz)
        l.addLayout(h)
        
        btn = QPushButton("Auto-Suggest Sizes")
        btn.clicked.connect(self._auto_suggest_sizes)
        l.addWidget(btn)
        layout.addWidget(g)

    def _create_volume_tonnage_group(self, layout):
        """Create volume and tonnage calculation group."""
        g = QGroupBox("2.5 Volume & Tonnage Calculation")
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.SUCCESS}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")

        form_layout = QFormLayout(g)
        form_layout.setSpacing(8)

        # Volume calculation method
        volume_layout = QHBoxLayout()
        self.volume_method = QComboBox()
        self.volume_method.addItems([
            "From Block Dimensions (DX*DY*DZ)",
            "From Column",
            "Calculate from Formula"
        ])
        self.volume_method.setCurrentText("From Block Dimensions (DX*DY*DZ)")
        self.volume_method.currentIndexChanged.connect(self._on_volume_method_changed)
        volume_layout.addWidget(self.volume_method)

        self.volume_col = QComboBox()
        self.volume_col.setVisible(False)
        self.volume_col.setToolTip("Select volume column")
        volume_layout.addWidget(self.volume_col)

        form_layout.addRow("Volume Method:", volume_layout)

        # Density source
        density_layout = QHBoxLayout()
        self.density_source = QComboBox()
        self.density_source.addItems([
            "Constant Value",
            "From Column",
            "Calculate from SG"
        ])
        self.density_source.setCurrentText("Constant Value")
        self.density_source.currentIndexChanged.connect(self._on_density_source_changed)
        density_layout.addWidget(self.density_source)

        self.density_col = QComboBox()
        self.density_col.setVisible(False)
        self.density_col.setToolTip("Select density/specific gravity column")
        density_layout.addWidget(self.density_col)

        form_layout.addRow("Density Source:", density_layout)

        # Constant density value
        density_value_layout = QHBoxLayout()
        self.density_value = QDoubleSpinBox()
        self.density_value.setRange(0.1, 10.0)
        self.density_value.setValue(2.7)
        self.density_value.setDecimals(3)
        self.density_value.setSingleStep(0.1)
        self.density_value.setSuffix(" t/m³")
        self.density_value.setToolTip("Constant density value for tonnage calculation")
        density_value_layout.addWidget(self.density_value)

        self.density_unit = QComboBox()
        self.density_unit.addItems(["t/m³", "g/cm³", "kg/m³"])
        self.density_unit.setCurrentText("t/m³")
        self.density_unit.currentIndexChanged.connect(self._on_density_unit_changed)
        density_value_layout.addWidget(self.density_unit)

        form_layout.addRow("Density Value:", density_value_layout)

        # Volume preview
        self.volume_preview = QLabel("Volume per block: -- m³")
        self.volume_preview.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding: 5px; background-color: {ModernColors.CARD_BG}; border-radius: 3px;")
        form_layout.addRow("", self.volume_preview)

        # Tonnage preview
        self.tonnage_preview = QLabel("Tonnage per block: -- tonnes")
        self.tonnage_preview.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding: 5px; background-color: {ModernColors.CARD_BG}; border-radius: 3px;")
        form_layout.addRow("", self.tonnage_preview)

        # Update previews when dimensions change
        self.dx.valueChanged.connect(self._update_volume_tonnage_preview)
        self.dy.valueChanged.connect(self._update_volume_tonnage_preview)
        self.dz.valueChanged.connect(self._update_volume_tonnage_preview)
        self.density_value.valueChanged.connect(self._update_volume_tonnage_preview)
        self.density_unit.currentIndexChanged.connect(self._update_volume_tonnage_preview)

        layout.addWidget(g)

    def _create_limits_group(self, layout):
        g = QGroupBox("2. Constraints")
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.WARNING}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        v_lay = QVBoxLayout(g)
        
        f = QFormLayout()
        self.max_blk = QSpinBox()
        self.max_blk.setRange(1000, 10000000)
        self.max_blk.setValue(100000)
        self.max_blk.valueChanged.connect(self._update_preview)
        f.addRow("Max Blocks:", self.max_blk)
        v_lay.addLayout(f)
        
        # Drillhole extent clipping
        from PyQt6.QtWidgets import QCheckBox
        self.clip_to_drillholes = QCheckBox("Clip to Drillhole Extent")
        self.clip_to_drillholes.setChecked(True)
        self.clip_to_drillholes.setToolTip("Limit block model grid to the area covered by drillholes.\nPrevents building blocks outside the data coverage area.")
        self.clip_to_drillholes.stateChanged.connect(self._update_preview)
        v_lay.addWidget(self.clip_to_drillholes)
        
        # Buffer/padding for extent
        buf_lay = QHBoxLayout()
        buf_lay.addWidget(QLabel("Extent Buffer (%):"))
        self.extent_buffer = QSpinBox()
        self.extent_buffer.setRange(0, 50)
        self.extent_buffer.setValue(0)
        self.extent_buffer.setToolTip("Add percentage buffer around drillhole extent")
        self.extent_buffer.valueChanged.connect(self._update_preview)
        buf_lay.addWidget(self.extent_buffer)
        buf_lay.addStretch()
        v_lay.addLayout(buf_lay)
        
        # Drillhole extent info
        self.dh_extent_lbl = QLabel("Drillhole extent: Not loaded")
        self.dh_extent_lbl.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
        v_lay.addWidget(self.dh_extent_lbl)
        
        layout.addWidget(g)

    def _on_method_changed(self, index: int):
        """Handle evaluation method selection change."""
        method = self.method_combo.currentText()
        self.property_combo.clear()
        
        if method == "-- Select Method --":
            self.data_status_lbl.setText("Select an evaluation method to see available properties.")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
            return
        
        # Find properties for this method
        available = [p for p in self._available_properties if p[0] == method]
        
        if available:
            for method_name, prop_name, var_name, _ in available:
                display_name = f"{prop_name}" + (f" (Variance: {var_name})" if var_name else "")
                self.property_combo.addItem(display_name, (prop_name, var_name))
            self.data_status_lbl.setText(f"{len(available)} property(ies) available from {method}")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
        else:
            self.data_status_lbl.setText(f"No results found for {method}. Run the analysis first.")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 10px;")
    
    def _on_property_changed(self, index: int):
        """Handle property selection change."""
        pass  # Property info will be used when loading data
    
    def _load_data_from_method(self):
        """Load estimation data from the selected method."""
        method = self.method_combo.currentText()
        
        if method == "-- Select Method --":
            QMessageBox.warning(self, "No Method Selected", "Please select an evaluation method first.")
            return
        
        if self.property_combo.count() == 0:
            QMessageBox.warning(
                self, "No Data Available",
                f"No results found for {method}.\n\n"
                "Please run the appropriate estimation/simulation first:\n"
                "• Go to Estimation menu\n"
                "• Select the desired method\n"
                "• Configure parameters and run\n"
                "• Then return here to build the block model."
            )
            return
        
        # Get selected property
        prop_data = self.property_combo.currentData()
        if not prop_data:
            return
        
        prop_name, var_name = prop_data
        
        # Get the results from registry
        results = self._kriging_results.get(method)
        if not results:
            QMessageBox.warning(self, "Data Not Found", f"Results for {method} not found in data registry.")
            return
        
        try:
            # Extract coordinates and values
            grid_x = results.get('grid_x')
            grid_y = results.get('grid_y')
            grid_z = results.get('grid_z')
            estimates = results.get('estimates')
            variances = results.get('variances')
            
            if grid_x is None or estimates is None:
                QMessageBox.warning(self, "Invalid Data", "Results are missing grid coordinates or estimates.")
                return
            
            # CRITICAL: Check array shapes and handle 3D vs 1D properly
            logger.info(f"Array shapes - grid_x: {grid_x.shape}, estimates: {estimates.shape}")
            
            # If estimates is 3D (same shape as grid), flatten both with same order
            # If estimates is 1D, it's already flattened - we need to match its order
            if estimates.ndim == 3:
                # Both are 3D - use same flatten order
                x = grid_x.ravel(order='F')
                y = grid_y.ravel(order='F')
                z = grid_z.ravel(order='F')
                v = estimates.ravel(order='F')
                logger.info("Using F-order for 3D estimates")
            else:
                # estimates is 1D - it was stored in F-order from kriging
                # grid_x/y/z are 3D - need to flatten with F-order to match
                x = grid_x.ravel(order='F')
                y = grid_y.ravel(order='F')
                z = grid_z.ravel(order='F')
                v = estimates  # Already 1D, keep as-is
                logger.info("Using F-order for 3D grids with 1D estimates")
            
            # Create DataFrame with proper column names
            df = pd.DataFrame({
                'X': x,
                'Y': y,
                'Z': z,
                prop_name: v
            })
            
            # Add variance if available
            actual_var_col = None
            if variances is not None and var_name:
                var_flat = variances.ravel()  # Use default C-order to match coordinates
                df[var_name] = var_flat
                actual_var_col = var_name

            # Sanity check grades before accepting
            finite_vals = df[prop_name][np.isfinite(df[prop_name])]
            if len(finite_vals) == 0:
                QMessageBox.critical(self, "Load Error", "Loaded estimates are all non-finite.")
                return
            gmin, gmax = float(finite_vals.min()), float(finite_vals.max())
            if max(abs(gmin), abs(gmax)) > 1e6:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Loaded estimates are extreme (min={gmin:.3e}, max={gmax:.3e}). "
                    "Please rerun kriging with correct parameters."
                )
                logger.warning(f"Blocked loading due to extreme grades: min={gmin}, max={gmax}")
                return
            
            # Set the data
            self.set_estimation_data(df, prop_name, actual_var_col)
            
            self.log.append(f"Loaded {len(df)} points from {method}")
            self.log.append(f"  Estimate Property: {prop_name}")
            self.log.append(f"    Range: [{v.min():.4f}, {v.max():.4f}]")
            if actual_var_col:
                var_vals = df[actual_var_col].values
                self.log.append(f"  Variance Property: {actual_var_col}")
                self.log.append(f"    Range: [{var_vals.min():.4f}, {var_vals.max():.4f}]")
            else:
                self.log.append(f"  Variance: Not available")
            
            self.data_status_lbl.setText(f"✓ Loaded {len(df):,} points from {method} ({prop_name})")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
            
        except Exception as e:
            logger.error(f"Failed to load data from {method}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{e}")
    
    def _refresh_available_data(self):
        """Refresh available estimation/simulation data from registry."""
        self._available_properties.clear()
        self._kriging_results.clear()
        
        if not self.registry:
            self.data_status_lbl.setText("Registry not available")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 10px;")
            return
        
        methods_found = []
        
        # Check for Ordinary Kriging results
        try:
            ok_results = self.registry.get_kriging_results()
            if ok_results:
                variable = ok_results.get('variable', 'Grade')
                prop_name = ok_results.get('property_name') or f"{variable}_est"
                var_name = ok_results.get('variance_property') or ok_results.get('variance_name') or 'Variance'
                self._kriging_results["Ordinary Kriging"] = ok_results
                self._available_properties.append(("Ordinary Kriging", prop_name, var_name, "kriging"))
                methods_found.append("Ordinary Kriging")
        except Exception as e:
            logger.debug(f"No OK results: {e}")
        
        # Check for Simple Kriging results
        try:
            if hasattr(self.registry, 'get_simple_kriging_results'):
                sk_results = self.registry.get_simple_kriging_results()
                if sk_results:
                    variable = sk_results.get('variable', 'Grade')
                    prop_name = sk_results.get('property_name') or f"{variable}_SK_est"
                    var_name = sk_results.get('variance_property') or sk_results.get('variance_name') or f"{variable}_SK_var"
                    self._kriging_results["Simple Kriging"] = sk_results
                    self._available_properties.append(("Simple Kriging", prop_name, var_name, "simple_kriging"))
                    methods_found.append("Simple Kriging")
        except Exception as e:
            logger.debug(f"No SK results: {e}")
        
        # Check for Universal Kriging results  
        try:
            if hasattr(self.registry, 'get_universal_kriging_results'):
                uk_results = self.registry.get_universal_kriging_results()
                if uk_results:
                    variable = uk_results.get('variable', 'Grade')
                    prop_name = uk_results.get('property_name') or f"{variable}_UK_est"
                    var_name = uk_results.get('variance_property') or uk_results.get('variance_name') or f"{variable}_UK_var"
                    self._kriging_results["Universal Kriging"] = uk_results
                    self._available_properties.append(("Universal Kriging", prop_name, var_name, "universal_kriging"))
                    methods_found.append("Universal Kriging")
        except Exception as e:
            logger.debug(f"No UK results: {e}")
        
        # Check for Co-Kriging results
        try:
            if hasattr(self.registry, 'get_cokriging_results'):
                ck_results = self.registry.get_cokriging_results()
                if ck_results:
                    variable = ck_results.get('primary_variable', 'Grade')
                    prop_name = ck_results.get('property_name') or f"{variable}_CK_est"
                    var_name = ck_results.get('variance_property') or ck_results.get('variance_name') or f"{variable}_CK_var"
                    self._kriging_results["Co-Kriging"] = ck_results
                    self._available_properties.append(("Co-Kriging", prop_name, var_name, "cokriging"))
                    methods_found.append("Co-Kriging")
        except Exception as e:
            logger.debug(f"No CK results: {e}")
        
        # Check for Indicator Kriging results
        try:
            if hasattr(self.registry, 'get_indicator_kriging_results'):
                ik_results = self.registry.get_indicator_kriging_results()
                if ik_results:
                    variable = ik_results.get('variable', 'Grade')
                    prop_name = ik_results.get('property_name') or f"{variable}_IK_mean"
                    var_name = ik_results.get('variance_property') or ik_results.get('variance_name') or None  # IK has probabilities, not variance
                    self._kriging_results["Indicator Kriging"] = ik_results
                    self._available_properties.append(("Indicator Kriging", prop_name, var_name, "indicator_kriging"))
                    methods_found.append("Indicator Kriging")
        except Exception as e:
            logger.debug(f"No IK results: {e}")
        
        # Check for SGSIM results
        try:
            if hasattr(self.registry, 'get_sgsim_results'):
                sgsim_results = self.registry.get_sgsim_results()
                if sgsim_results:
                    variable = sgsim_results.get('variable', 'Grade')
                    # SGSIM typically has mean and variance of realizations
                    prop_name = sgsim_results.get('property_name') or f"{variable}_SGSIM_mean"
                    var_name = sgsim_results.get('variance_property') or sgsim_results.get('variance_name') or f"{variable}_SGSIM_var"
                    self._kriging_results["SGSIM (Simulation)"] = sgsim_results
                    self._available_properties.append(("SGSIM (Simulation)", prop_name, var_name, "sgsim"))
                    methods_found.append("SGSIM")
        except Exception as e:
            logger.debug(f"No SGSIM results: {e}")
        
        # Update UI
        if methods_found:
            self.data_status_lbl.setText(f"Found results: {', '.join(methods_found)}")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
            self.log.append(f"Refreshed data sources: {', '.join(methods_found)}")
        else:
            self.data_status_lbl.setText("No estimation results found. Run an estimation method first.")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 10px;")
        
        # Refresh the current method's property list
        self._on_method_changed(self.method_combo.currentIndex())

    def _on_krig_loaded(self, res):
        """Handle kriging results loaded from registry - store for later use."""
        try:
            # Extract the variable name from results
            variable = res.get('variable', 'Grade')
            
            # Store results with proper property names
            self._kriging_results["Ordinary Kriging"] = res
            prop_name = res.get('property_name') or f"{variable}_est"
            var_name = res.get('variance_property') or res.get('variance_name') or 'Variance'
            
            # Add to available properties if not already there
            existing = [p for p in self._available_properties if p[0] == "Ordinary Kriging"]
            if not existing:
                self._available_properties.append(("Ordinary Kriging", prop_name, var_name, "kriging"))
            
            # Auto-select Ordinary Kriging in combo if nothing selected
            if self.method_combo.currentText() == "-- Select Method --":
                idx = self.method_combo.findText("Ordinary Kriging")
                if idx >= 0:
                    self.method_combo.setCurrentIndex(idx)
            
            self.log.append(f"Kriging results available: {variable}")
            logger.info(f"Kriging results loaded for variable: {variable}")
            
        except Exception as e:
            logger.warning(f"Failed to load kriging results: {e}")

    def _on_sgsim_loaded(self, res):
        """Handle SGSIM results loaded from registry - store for later use."""
        try:
            variable = res.get('variable', 'Grade')
            
            # Store results
            self._kriging_results["SGSIM (Simulation)"] = res
            prop_name = f"{variable}_SGSIM_mean"
            var_name = f"{variable}_SGSIM_var"
            
            # Add to available properties if not already there
            existing = [p for p in self._available_properties if p[0] == "SGSIM (Simulation)"]
            if not existing:
                self._available_properties.append(("SGSIM (Simulation)", prop_name, var_name, "sgsim"))
            
            # Refresh property combo if SGSIM is selected
            if self.method_combo.currentText() == "SGSIM (Simulation)":
                self._on_method_changed(self.method_combo.currentIndex())
            
            self.log.append(f"SGSIM results available: {variable}")
            logger.info(f"SGSIM results loaded for variable: {variable}")
            
        except Exception as e:
            logger.warning(f"Failed to load SGSIM results: {e}")

    def _on_simple_kriging_loaded(self, res):
        """Handle Simple Kriging results loaded from registry."""
        try:
            variable = res.get('variable', 'Grade')
            self._kriging_results["Simple Kriging"] = res
            prop_name = f"{variable}_SK_est"
            var_name = f"{variable}_SK_var"
            
            existing = [p for p in self._available_properties if p[0] == "Simple Kriging"]
            if not existing:
                self._available_properties.append(("Simple Kriging", prop_name, var_name, "simple_kriging"))
            
            if self.method_combo.currentText() == "Simple Kriging":
                self._on_method_changed(self.method_combo.currentIndex())
            
            self.log.append(f"Simple Kriging results available: {variable}")
        except Exception as e:
            logger.warning(f"Failed to load Simple Kriging results: {e}")

    def _on_universal_kriging_loaded(self, res):
        """Handle Universal Kriging results loaded from registry."""
        try:
            variable = res.get('variable', 'Grade')
            self._kriging_results["Universal Kriging"] = res
            prop_name = f"{variable}_UK_est"
            var_name = f"{variable}_UK_var"
            
            existing = [p for p in self._available_properties if p[0] == "Universal Kriging"]
            if not existing:
                self._available_properties.append(("Universal Kriging", prop_name, var_name, "universal_kriging"))
            
            if self.method_combo.currentText() == "Universal Kriging":
                self._on_method_changed(self.method_combo.currentIndex())
            
            self.log.append(f"Universal Kriging results available: {variable}")
        except Exception as e:
            logger.warning(f"Failed to load Universal Kriging results: {e}")

    def _on_cokriging_loaded(self, res):
        """Handle Co-Kriging results loaded from registry."""
        try:
            variable = res.get('primary_variable', res.get('variable', 'Grade'))
            self._kriging_results["Co-Kriging"] = res
            prop_name = f"{variable}_CK_est"
            var_name = f"{variable}_CK_var"
            
            existing = [p for p in self._available_properties if p[0] == "Co-Kriging"]
            if not existing:
                self._available_properties.append(("Co-Kriging", prop_name, var_name, "cokriging"))
            
            if self.method_combo.currentText() == "Co-Kriging":
                self._on_method_changed(self.method_combo.currentIndex())
            
            self.log.append(f"Co-Kriging results available: {variable}")
        except Exception as e:
            logger.warning(f"Failed to load Co-Kriging results: {e}")

    def _on_indicator_kriging_loaded(self, res):
        """Handle Indicator Kriging results loaded from registry."""
        try:
            variable = res.get('variable', 'Grade')
            self._kriging_results["Indicator Kriging"] = res
            prop_name = f"{variable}_IK_mean"
            var_name = None  # IK uses probabilities
            
            existing = [p for p in self._available_properties if p[0] == "Indicator Kriging"]
            if not existing:
                self._available_properties.append(("Indicator Kriging", prop_name, var_name, "indicator_kriging"))
            
            if self.method_combo.currentText() == "Indicator Kriging":
                self._on_method_changed(self.method_combo.currentIndex())
            
            self.log.append(f"Indicator Kriging results available: {variable}")
        except Exception as e:
            logger.warning(f"Failed to load Indicator Kriging results: {e}")

    def _on_rbf_loaded(self, res):
        """Handle RBF Interpolation results loaded from registry."""
        try:
            variable = res.get('metadata', {}).get('variable', 'Grade')
            self._kriging_results["RBF Interpolation"] = res
            prop_name = f"{variable}_RBF"
            var_name = None  # RBF produces single property

            existing = [p for p in self._available_properties if p[0] == "RBF Interpolation"]
            if not existing:
                self._available_properties.append(("RBF Interpolation", prop_name, var_name, "rbf"))

            if self.method_combo.currentText() == "RBF Interpolation":
                self._on_method_changed(self.method_combo.currentIndex())

            self.log.append(f"RBF Interpolation results available: {variable}")
        except Exception as e:
            logger.warning(f"Failed to load RBF Interpolation results: {e}")

    def _on_drillhole_loaded(self, data):
        """Load drillhole data and compute extents for grid clipping."""
        try:
            df = None
            if isinstance(data, dict):
                # Prefer composites over raw assays
                composites = data.get('composites')
                assays = data.get('assays')
                if isinstance(composites, pd.DataFrame) and not composites.empty:
                    df = composites
                elif isinstance(assays, pd.DataFrame) and not assays.empty:
                    df = assays
            elif isinstance(data, pd.DataFrame):
                df = data
            
            if df is None or df.empty:
                logger.warning("Block Model Builder: No drillhole data available")
                return
            
            # Ensure X, Y, Z columns exist
            from ..utils.coordinate_utils import ensure_xyz_columns
            df = ensure_xyz_columns(df)
            
            if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
                logger.warning("Block Model Builder: Drillhole data missing X, Y, Z coordinates")
                return
            
            self.drillhole_data = df
            
            # Compute extents
            self.drillhole_extents = {
                'xmin': df['X'].min(),
                'xmax': df['X'].max(),
                'ymin': df['Y'].min(),
                'ymax': df['Y'].max(),
                'zmin': df['Z'].min(),
                'zmax': df['Z'].max(),
            }
            
            # Update label
            ext = self.drillhole_extents
            self.dh_extent_lbl.setText(
                f"Drillhole extent: X[{ext['xmin']:.0f}-{ext['xmax']:.0f}] "
                f"Y[{ext['ymin']:.0f}-{ext['ymax']:.0f}] "
                f"Z[{ext['zmin']:.0f}-{ext['zmax']:.0f}]"
            )
            self.dh_extent_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
            
            logger.info(f"Block Model Builder: Loaded drillhole extents from {len(df)} samples")
            self._update_preview()
            
        except Exception as e:
            logger.warning(f"Block Model Builder: Failed to load drillhole extents: {e}")

    def set_estimation_data(self, df, grade_col, var_col=None):
        """Set estimation data for block model building.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with X, Y, Z coordinates and grade/variance columns
        grade_col : str
            Column name for the grade/estimate values
        var_col : str, optional
            Column name for variance values
        """
        self.estimation_data = df
        self.grade_col = grade_col
        self.var_col = var_col
        self.build_btn.setEnabled(True)
        self._update_preview()
        
        # Update data status label
        if hasattr(self, 'data_status_lbl'):
            grade_range = f"[{df[grade_col].min():.4f}, {df[grade_col].max():.4f}]" if grade_col in df.columns else "N/A"
            self.data_status_lbl.setText(f"✓ Data loaded: {len(df):,} points, Property: {grade_col}, Range: {grade_range}")
            self.data_status_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
        
        # Update volume and density column options
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.volume_col.clear()
        self.volume_col.addItems(numeric_cols)

        self.density_col.clear()
        self.density_col.addItems(numeric_cols)

        # Update preview calculations
        self._update_volume_tonnage_preview()

        self.log.append(f"Loaded {len(df)} points.")
        self.log.append(f"  Property: {grade_col}")
        if var_col:
            self.log.append(f"  Variance: {var_col}")

    def _update_preview(self):
        if self.estimation_data is None:
            self.preview_lbl.setText("Load estimation data to see preview.")
            return
        
        # Extent calc
        dx, dy, dz = self.dx.value(), self.dy.value(), self.dz.value()
        
        # Get extents - use drillhole extents if clipping enabled
        clip_enabled = hasattr(self, 'clip_to_drillholes') and self.clip_to_drillholes.isChecked()
        
        if clip_enabled and self.drillhole_extents is not None:
            # Use drillhole extents with buffer
            buffer_pct = self.extent_buffer.value() / 100.0
            ext = self.drillhole_extents
            
            x_range = ext['xmax'] - ext['xmin']
            y_range = ext['ymax'] - ext['ymin']
            z_range = ext['zmax'] - ext['zmin']
            
            # Add buffer
            x_r = x_range * (1 + 2 * buffer_pct)
            y_r = y_range * (1 + 2 * buffer_pct)
            z_r = z_range * (1 + 2 * buffer_pct)
            
            extent_source = "Drillhole"
        else:
            # Use estimation data extents
            x_r = self.estimation_data['X'].max() - self.estimation_data['X'].min()
            y_r = self.estimation_data['Y'].max() - self.estimation_data['Y'].min()
            z_r = self.estimation_data['Z'].max() - self.estimation_data['Z'].min()
            extent_source = "Estimation"
        
        nx = max(1, int(np.ceil(x_r / dx)))
        ny = max(1, int(np.ceil(y_r / dy)))
        nz = max(1, int(np.ceil(z_r / dz)))
        total = nx * ny * nz
        
        txt = f"Dimensions: {nx} x {ny} x {nz}\nTotal Blocks: {total:,}"
        txt += f"\n\nExtent Source: {extent_source}"
        
        if clip_enabled and self.drillhole_extents is None:
            txt += "\n\n⚠️ No drillhole data - using estimation extent"
        
        if total > self.max_blk.value():
            txt += "\n\nWARNING: Exceeds Max Blocks!"
        
        self.preview_lbl.setText(txt)

    def _auto_suggest_sizes(self):
        if self.estimation_data is None:
            return
        # Calculate suggested sizes based on data spacing
        try:
            x_coords = self.estimation_data['X'].values
            y_coords = self.estimation_data['Y'].values
            z_coords = self.estimation_data['Z'].values
            
            # Calculate median spacing
            if len(x_coords) > 1:
                x_spacing = np.median(np.diff(np.sort(np.unique(x_coords))))
                y_spacing = np.median(np.diff(np.sort(np.unique(y_coords))))
                z_spacing = np.median(np.diff(np.sort(np.unique(z_coords))))
                
                # Suggest sizes (round to reasonable values)
                self.dx.setValue(max(1.0, round(x_spacing / 2, 1)))
                self.dy.setValue(max(1.0, round(y_spacing / 2, 1)))
                self.dz.setValue(max(1.0, round(z_spacing / 2, 1)))
            else:
                # Default suggestions
                self.dx.setValue(10)
                self.dy.setValue(10)
                self.dz.setValue(5)
        except Exception as e:
            logger.warning(f"Auto-suggest failed: {e}")
            # Fallback to defaults
            self.dx.setValue(10)
            self.dy.setValue(10)
            self.dz.setValue(5)

    def build_block_model(self):
        print("DEBUG: build_block_model() called")  # DEBUG
        print(f"DEBUG: estimation_data = {self.estimation_data is not None}")  # DEBUG
        print(f"DEBUG: controller = {hasattr(self, 'controller') and self.controller is not None}")  # DEBUG

        if self.estimation_data is None:
            print("DEBUG: No estimation data - showing warning")  # DEBUG
            QMessageBox.warning(self, "No Data", "Load estimation data first.")
            return

        # Check if controller is available (panel might not have it if not injected)
        if not hasattr(self, 'controller') or self.controller is None:
            print("DEBUG: No controller - showing warning")  # DEBUG
            # Fallback: try to get controller from parent or use direct call
            QMessageBox.warning(self, "Error", "Controller not available. This panel requires a controller instance.")
            return
        
        # Store the current method name for visualization
        self.current_method = self.method_combo.currentText()
        if self.current_method == "-- Select Method --":
            self.current_method = "Block Model"  # Fallback name
        
        self.log.append("Building...")
        self.build_btn.setEnabled(False)
        
        # Determine extents to use
        clip_enabled = hasattr(self, 'clip_to_drillholes') and self.clip_to_drillholes.isChecked()
        extents = None
        
        if clip_enabled and self.drillhole_extents is not None:
            buffer_pct = self.extent_buffer.value() / 100.0
            ext = self.drillhole_extents
            
            x_buf = (ext['xmax'] - ext['xmin']) * buffer_pct
            y_buf = (ext['ymax'] - ext['ymin']) * buffer_pct
            z_buf = (ext['zmax'] - ext['zmin']) * buffer_pct
            
            # Pass as tuple: (xmin, xmax, ymin, ymax, zmin, zmax)
            extents = (
                ext['xmin'] - x_buf,
                ext['xmax'] + x_buf,
                ext['ymin'] - y_buf,
                ext['ymax'] + y_buf,
                ext['zmin'] - z_buf,
                ext['zmax'] + z_buf,
            )
            self.log.append(f"Using drillhole extents with {self.extent_buffer.value()}% buffer")
            self.log.append(f"  X: {extents[0]:.1f} - {extents[1]:.1f}")
            self.log.append(f"  Y: {extents[2]:.1f} - {extents[3]:.1f}")
            self.log.append(f"  Z: {extents[4]:.1f} - {extents[5]:.1f}")
        else:
            self.log.append("Using estimation data extents")
        
        # Prepare params
        volume_tonnage_config = self._get_volume_tonnage_config()

        params = {
            'estimation_df': self.estimation_data.copy(),
            'xinc': self.dx.value(),
            'yinc': self.dy.value(),
            'zinc': self.dz.value(),
            'grade_col': self.grade_col,
            'var_col': self.var_col,
            'max_blocks': self.max_blk.value(),
            'extents': extents,  # Optional: drillhole-based extents
            'volume_tonnage_config': volume_tonnage_config,  # New: volume/tonnage calculation config
        }
        
        # Progress callback
        def progress_callback(percent: int, message: str):
            self.log.append(f"{percent}%: {message}")
        
        # Run via controller
        self.controller.run_task(
            'build_block_model',
            params,
            callback=self._on_build_complete,
            progress_callback=progress_callback
        )

    def _on_build_complete(self, result: Dict[str, Any]):
        """Handle completion of block model build task."""
        if result is None:
            self._on_error("Build returned no result.")
            return
        
        if result.get("error"):
            self._on_error(result["error"])
            return
        
        # Extract results
        df = result.get("block_df")
        grid_def = result.get("grid_def")  # Grid definition dict, not PyVista grid
        info = result.get("info")
        
        if df is None:
            self._on_error("No block DataFrame in result.")
            return
        
        # Store DataFrame and info
        self.block_model_df = df
        self.block_info = info
        
        # Create PyVista grid from grid_def in main thread (required for PyVista)
        try:
            import pyvista as pv
            if grid_def and isinstance(grid_def, dict):
                x_edges = grid_def.get('x_edges')
                y_edges = grid_def.get('y_edges')
                z_edges = grid_def.get('z_edges')
                
                if x_edges is not None and y_edges is not None and z_edges is not None:
                    # Create RectilinearGrid from edge coordinates
                    self.block_grid = pv.RectilinearGrid(x_edges, y_edges, z_edges)
                    
                    # Add cell data from info if available
                    if info and 'cell_data' in info:
                        cell_data = info['cell_data']
                        for prop_name, prop_values in cell_data.items():
                            if prop_values is not None:
                                # Validate array size matches grid cells
                                expected_cells = self.block_grid.n_cells
                                if len(prop_values) != expected_cells:
                                    logger.error(f"Cell data '{prop_name}' size mismatch: {len(prop_values)} vs {expected_cells} cells")
                                    continue
                                
                                # Log data range for validation
                                valid_vals = prop_values[np.isfinite(prop_values)]
                                if len(valid_vals) > 0:
                                    logger.info(f"Adding cell data '{prop_name}': range [{valid_vals.min():.4f}, {valid_vals.max():.4f}], valid={len(valid_vals)}/{len(prop_values)}")
                                    self.log.append(f"  {prop_name}: [{valid_vals.min():.4f}, {valid_vals.max():.4f}]")
                                else:
                                    logger.warning(f"Cell data '{prop_name}' has no valid values")
                                
                                self.block_grid[prop_name] = prop_values
                    
                    # Log all available properties
                    all_properties = list(self.block_grid.cell_data.keys())
                    logger.info(f"Created PyVista grid: {self.block_grid.n_cells} cells, properties: {all_properties}")
                    self.log.append(f"\nBlock model properties (available for visualization):")
                    for prop in all_properties:
                        self.log.append(f"  • {prop}")
                else:
                    logger.warning("Grid definition missing edge coordinates")
                    self.block_grid = None
            else:
                logger.warning("No grid_def in result")
                self.block_grid = None
        except Exception as e:
            logger.error(f"Failed to create PyVista grid: {e}", exc_info=True)
            self.block_grid = None
        
        self.log.append("Build Complete.")
        
        # Enable buttons if we have valid data
        if self.block_model_df is not None:
            self.viz_btn.setEnabled(True)
            self.exp_btn.setEnabled(True)
        self.build_btn.setEnabled(True)
        
        # Register to registry (UI thread - safe)
        # This makes the block model available to other panels like Resource Classification
        if hasattr(self, 'registry') and self.registry:
            try:
                self.registry.register_block_model_generated(df, source_panel="BlockModelBuilder")
                logger.info(f"Block Model Builder: Registered {len(df):,} blocks to DataRegistry")
            except AttributeError as e:
                logger.warning(f"register_block_model_generated not available in registry: {e}")
            except Exception as e:
                logger.error(f"Failed to register block model to registry: {e}", exc_info=True)
        else:
            # Try to get registry again in case it wasn't available at init
            logger.warning("Block Model Builder: Registry not available, attempting to resolve...")
            try:
                self.registry = self.get_registry()
                if self.registry:
                    self.registry.register_block_model_generated(df, source_panel="BlockModelBuilder")
                    logger.info(f"Block Model Builder: Registered {len(df):,} blocks to DataRegistry (late bind)")
                else:
                    logger.error("Block Model Builder: Could not resolve DataRegistry - block model not registered!")
            except Exception as e:
                logger.error(f"Block Model Builder: Failed to register block model: {e}", exc_info=True)

    def _on_error(self, error_msg: str):
        """Handle errors from build task."""
        QMessageBox.critical(self, "Build Error", f"Failed to build block model:\n{error_msg}")
        self.log.append(f"ERROR: {error_msg}")
        self.build_btn.setEnabled(True)

    def visualize_block_model(self):
        # Check if we have a valid block model (either grid or DataFrame)
        if self.block_model_df is None:
            QMessageBox.warning(self, "No Model", "Build block model first.")
            return
        
        # Get method name for layer naming (fallback to stored value or default)
        method_name = self.current_method if hasattr(self, 'current_method') and self.current_method else "Block Model"
        
        # Prefer grid if available, otherwise use DataFrame
        if self.block_grid is not None:
            self.request_visualization.emit(self.block_grid, self.grade_col, method_name)
        elif self.block_model_df is not None:
            # Fallback: emit DataFrame if grid creation failed
            logger.warning("Visualizing using DataFrame (grid creation may have failed)")
            self.request_visualization.emit(self.block_model_df, self.grade_col, method_name)
        else:
            QMessageBox.warning(self, "No Model", "Build block model first.")

    def export_block_model(self):
        if self.block_model_df is None:
            QMessageBox.warning(self, "No Model", "Build block model first.")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export", "block_model.csv", "*.csv")
        if f:
            try:
                self.block_model_df.to_csv(f, index=False)
                QMessageBox.information(self, "Export", f"Block model exported to {f}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export: {e}")

    # ============================================================================
    # VOLUME & TONNAGE CALCULATION METHODS
    # ============================================================================

    def _on_volume_method_changed(self, index: int):
        """Handle volume calculation method change."""
        method = self.volume_method.currentText()
        show_volume_col = (method == "From Column")

        self.volume_col.setVisible(show_volume_col)
        self._update_volume_tonnage_preview()

        # Update available columns for volume
        if show_volume_col and self.estimation_data is not None:
            numeric_cols = self.estimation_data.select_dtypes(include=[np.number]).columns.tolist()
            self.volume_col.clear()
            self.volume_col.addItems(numeric_cols)

    def _on_density_source_changed(self, index: int):
        """Handle density source change."""
        source = self.density_source.currentText()
        show_density_col = (source == "From Column")

        self.density_col.setVisible(show_density_col)
        self.density_value.setEnabled(source == "Constant Value")

        # Update available columns for density
        if show_density_col and self.estimation_data is not None:
            numeric_cols = self.estimation_data.select_dtypes(include=[np.number]).columns.tolist()
            self.density_col.clear()
            self.density_col.addItems(numeric_cols)

        self._update_volume_tonnage_preview()

    def _on_density_unit_changed(self, index: int):
        """Handle density unit change."""
        unit = self.density_unit.currentText()
        current_value = self.density_value.value()

        # Convert value based on unit
        if unit == "g/cm³":
            # Convert from t/m³ to g/cm³
            converted = current_value * 1000 / 1e6
            self.density_value.setSuffix(" g/cm³")
            self.density_value.setRange(0.1, 10.0)
            self.density_value.setSingleStep(0.1)
        elif unit == "kg/m³":
            # Convert from t/m³ to kg/m³
            converted = current_value * 1000
            self.density_value.setSuffix(" kg/m³")
            self.density_value.setRange(100, 10000)
            self.density_value.setSingleStep(100)
        else:  # t/m³
            # Keep as t/m³
            converted = current_value
            self.density_value.setSuffix(" t/m³")
            self.density_value.setRange(0.1, 10.0)
            self.density_value.setSingleStep(0.1)

        self.density_value.blockSignals(True)
        self.density_value.setValue(converted)
        self.density_value.blockSignals(False)

        self._update_volume_tonnage_preview()

    def _update_volume_tonnage_preview(self):
        """Update volume and tonnage preview calculations."""
        if not hasattr(self, 'dx') or not hasattr(self, 'volume_preview'):
            return

        try:
            # Calculate volume
            dx = self.dx.value()
            dy = self.dy.value()
            dz = self.dz.value()

            if dx > 0 and dy > 0 and dz > 0:
                volume = dx * dy * dz
                self.volume_preview.setText(f"Volume per block: {volume:.2f} m³")
            else:
                self.volume_preview.setText("Volume per block: -- m³")

            # Calculate tonnage
            density_value = self.density_value.value()
            unit = self.density_unit.currentText()

            # Convert to t/m³ for calculation
            if unit == "g/cm³":
                density_t_m3 = density_value * 1000  # g/cm³ to kg/m³, then to t/m³
            elif unit == "kg/m³":
                density_t_m3 = density_value / 1000  # kg/m³ to t/m³
            else:  # already t/m³
                density_t_m3 = density_value

            if volume > 0 and density_t_m3 > 0:
                tonnage = volume * density_t_m3
                self.tonnage_preview.setText(f"Tonnage per block: {tonnage:.2f} tonnes")
            else:
                self.tonnage_preview.setText("Tonnage per block: -- tonnes")

        except Exception as e:
            logger.warning(f"Error updating volume/tonnage preview: {e}")
            self.volume_preview.setText("Volume per block: -- m³")
            self.tonnage_preview.setText("Tonnage per block: -- tonnes")

    def _get_volume_tonnage_config(self) -> Dict[str, Any]:
        """Get volume and tonnage calculation configuration."""
        config = {
            'volume_method': self.volume_method.currentText(),
            'volume_column': self.volume_col.currentText() if self.volume_col.isVisible() else None,
            'density_source': self.density_source.currentText(),
            'density_column': self.density_col.currentText() if self.density_col.isVisible() else None,
            'density_value': self.density_value.value(),
            'density_unit': self.density_unit.currentText(),
        }
        return config

    def refresh_theme(self):
        """Refresh all UI elements to match current theme."""
        # Build log
        if hasattr(self, 'log'):
            self.log.setStyleSheet(f"background-color: {ModernColors.ELEVATED_BG}; color: {ModernColors.TEXT_PRIMARY}; font-family: Consolas;")

        # Build button
        if hasattr(self, 'build_btn'):
            self.build_btn.setStyleSheet(f"background-color: {ModernColors.SUCCESS}; color: white; font-weight: bold; padding: 12px;")

        # Load data button
        if hasattr(self, 'load_data_btn'):
            self.load_data_btn.setStyleSheet(f"background-color: {ModernColors.INFO}; color: white; padding: 6px;")

        # Status labels
        if hasattr(self, 'data_status_lbl'):
            # Preserve current text, just update colors based on current state
            current_text = self.data_status_lbl.text()
            if "✓" in current_text or "Found results:" in current_text:
                self.data_status_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")
            elif "No results" in current_text or "Select an evaluation" in current_text:
                self.data_status_lbl.setStyleSheet(f"color: {ModernColors.WARNING}; font-size: 10px;")
            elif "Registry not available" in current_text:
                self.data_status_lbl.setStyleSheet(f"color: {ModernColors.ERROR}; font-size: 10px;")
            else:
                self.data_status_lbl.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")

        if hasattr(self, 'dh_extent_lbl'):
            current_text = self.dh_extent_lbl.text()
            if "Not loaded" in current_text:
                self.dh_extent_lbl.setStyleSheet(f"color: {ModernColors.TEXT_HINT}; font-size: 10px;")
            else:
                self.dh_extent_lbl.setStyleSheet(f"color: {ModernColors.SUCCESS}; font-size: 10px;")

        # Volume and tonnage previews
        if hasattr(self, 'volume_preview'):
            self.volume_preview.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding: 5px; background-color: {ModernColors.CARD_BG}; border-radius: 3px;")

        if hasattr(self, 'tonnage_preview'):
            self.tonnage_preview.setStyleSheet(f"color: {ModernColors.TEXT_SECONDARY}; font-size: 10px; padding: 5px; background-color: {ModernColors.CARD_BG}; border-radius: 3px;")

        # Group boxes - need to find them and refresh their stylesheets
        for child in self.findChildren(QGroupBox):
            title = child.title()
            if "Data Source" in title:
                child.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.SUCCESS}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
            elif "Block Dimensions" in title:
                child.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.INFO}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
            elif "Volume & Tonnage" in title:
                child.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.SUCCESS}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
            elif "Constraints" in title:
                child.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: {ModernColors.WARNING}; border: 1px solid {ModernColors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
