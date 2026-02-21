"""
JORC/SAMREC Resource Classification Panel (Redesigned - VERTICAL STACK V2)
==========================================================================
Updated: 2025-12-01 (Vertical Layout Fix)
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox, 
    QSpinBox, QComboBox, QPushButton, QLabel, QMessageBox, 
    QWidget, QFrame, QProgressBar, QTableWidget, QTableWidgetItem, 
    QHeaderView, QSlider, QSplitter, QScrollArea, QSizePolicy, 
    QApplication, QToolButton
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QFont

from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
from ..models.jorc_classification_engine import (
    JORCClassificationEngine,
    VariogramModel,
    ClassificationRuleset,
    ClassificationResult,
    CLASSIFICATION_COLORS,
    CLASSIFICATION_ORDER,
)
from ..utils.coordinate_utils import ensure_xyz_columns

logger = logging.getLogger(__name__)

# Worker logic moved to MiningController._prepare_resource_classification_payload
# This ensures pure computation with no access to DataRegistry or Qt objects

class ModernSlider(QWidget):
    """
    A combined Slider + SpinBox for precise control.
    Layout: [Label] [Slider -----------] [SpinBox %] [Real Value Label]
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, label: str, value: int, color: str, parent=None):
        super().__init__(parent)
        self.range_major = 100.0
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Row 1: Label and Value
        top_row = QHBoxLayout()
        self.lbl_title = QLabel(label)
        self.lbl_title.setStyleSheet(f"color: {ModernColors.TEXT_PRIMARY}; font-weight: 500;")
        top_row.addWidget(self.lbl_title)
        
        top_row.addStretch()
        
        self.lbl_real_value = QLabel("= 0.0 m")
        self.lbl_real_value.setStyleSheet("color: #909090; font-family: monospace;")
        top_row.addWidget(self.lbl_real_value)
        layout.addLayout(top_row)
        
        # Row 2: Slider and Spinbox
        ctrl_row = QHBoxLayout()
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.slider.setRange(0, 300)
        self.slider.setValue(value)
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: #303038;
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: white;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
        """)
        
        self.spin = QSpinBox()
        self.spin.setRange(0, 300)
        self.spin.setValue(value)
        self.spin.setSuffix("%")
        self.spin.setFixedWidth(70)
        self.spin.setStyleSheet(f"border: 1px solid #404040; background: #202020; color: {color}; font-weight: bold;")
        
        # Sync controls
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._on_change)
        
        ctrl_row.addWidget(self.slider)
        ctrl_row.addWidget(self.spin)
        layout.addLayout(ctrl_row)
        
        self._on_change(value)



    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
        
                    CategoryCard {{
                        background-color: #1a1a20;
                        border: 1px solid #303038;
                        border-left: 5px solid {color};
                        border-radius: 4px;
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
    def _on_change(self, val):
        real_dist = (val / 100.0) * self.range_major
        self.lbl_real_value.setText(f"= {real_dist:.1f} m")
        self.valueChanged.emit(val)
        
    def set_variogram_range(self, r):
        self.range_major = r
        self._on_change(self.slider.value())
        
    def value(self):
        return self.slider.value()

class CategoryCard(QFrame):
    """
    Full-width card for a classification category.
    """
    def __init__(self, name, color, default_dist, default_holes):
        super().__init__()
        self.name = name
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # CRITICAL: Set size policy to expand horizontally
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(300)  # Ensure minimum width
        self.setStyleSheet(self._get_stylesheet())
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        
        # Header
        header = QHBoxLayout()
        lbl_name = QLabel(name)
        lbl_name.setStyleSheet(f"color: {color}; font-size: 11pt; font-weight: bold;")
        header.addWidget(lbl_name)
        layout.addLayout(header)
        
        # Controls
        self.dist_slider = ModernSlider("Max Distance (% of Range)", default_dist, color)
        layout.addWidget(self.dist_slider)
        
        row_holes = QHBoxLayout()
        row_holes.addWidget(QLabel("Min Unique Holes:"))
        self.spin_holes = QSpinBox()
        self.spin_holes.setRange(1, 20)
        self.spin_holes.setValue(default_holes)
        self.spin_holes.setFixedWidth(60)
        row_holes.addWidget(self.spin_holes)
        row_holes.addStretch()
        layout.addLayout(row_holes)

    def set_variogram_range(self, r):
        self.dist_slider.set_variogram_range(r)
        
    def get_params(self):
        return {
            "dist_pct": self.dist_slider.value(),
            "min_holes": self.spin_holes.value()
        }

class JORCClassificationPanel(BaseAnalysisPanel):
    task_name = "jorc_classification"
    classification_complete = pyqtSignal(object)
    request_visualization = pyqtSignal(object, str) # Added signal for visualization
    
    def __init__(self, parent=None):
        self.drillhole_data = None
        self.block_model_data = None
        self.classification_result = None
        self.cards = {}

        # Storage for different block model sources
        self._block_model_sources: Dict[str, Any] = {}  # name -> data
        self._current_source: str = ""
        self._available_sources: list = []

        super().__init__(parent=parent, panel_id="jorc_classification")
        self.setWindowTitle("Resource Classification")
        self._apply_theme()
        
        # Build UI
        self._build_ui()
        
        # Init Registry (connect to data signals and load existing data)
        self._init_registry()
    
    def _build_ui(self):
        """Build the UI layout."""
        # UI Construction
        # Clear any existing layout from base class
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
        
        # Create new layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)
        
        # 1. Top Bar
        self.main_layout.addWidget(self._create_header())
        
        # 2. Main Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #303038; width: 2px; }")
        
        # LEFT: Configuration (Scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        config_widget = QWidget()
        config_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        config_layout = QVBoxLayout(config_widget)
        # FIXED: Reduced margins and spacing to eliminate wasted space
        config_layout.setContentsMargins(15, 15, 15, 15)
        config_layout.setSpacing(12)

        # Block Model Source Section
        source_group = QGroupBox("Block Model Source")
        source_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold; color: #4da6ff;
                border: 1px solid #404050; margin-top: 6px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        source_layout = QVBoxLayout(source_group)

        # Data source combo box
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Select Block Model:"))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(250)
        self.source_combo.addItem("No block model loaded", "none")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_combo.setToolTip("Select which block model to use for classification")
        source_row.addWidget(self.source_combo, stretch=1)
        source_layout.addLayout(source_row)

        # Current source info label
        self.source_info_label = QLabel("No block model selected")
        self.source_info_label.setStyleSheet("color: #888; font-style: italic; padding: 5px;")
        source_layout.addWidget(self.source_info_label)

        config_layout.addWidget(source_group)

        # Variogram Section
        var_group = QGroupBox("Variogram Parameters")
        var_layout = QHBoxLayout(var_group)
        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(1, 10000)
        self.spin_range.setValue(100)
        self.spin_range.setPrefix("Major Range: ")
        self.spin_range.setSuffix(" m")
        self.spin_range.valueChanged.connect(self._on_var_changed)
        
        self.spin_sill = QDoubleSpinBox()
        self.spin_sill.setValue(1.0)
        self.spin_sill.setPrefix("Sill: ")
        
        var_layout.addWidget(self.spin_range)
        var_layout.addWidget(self.spin_sill)
        config_layout.addWidget(var_group)
        
        # Cards (Stacked Vertically - ONE PER ROW, FULL WIDTH)
        # CRITICAL: Using QVBoxLayout.addWidget() stacks widgets vertically
        self.cards["Measured"] = CategoryCard("Measured", CLASSIFICATION_COLORS["Measured"], 25, 3)
        self.cards["Indicated"] = CategoryCard("Indicated", CLASSIFICATION_COLORS["Indicated"], 60, 2)
        self.cards["Inferred"] = CategoryCard("Inferred", CLASSIFICATION_COLORS["Inferred"], 150, 1)
        
        # Add cards one by one - each takes full width of the vertical layout
        config_layout.addWidget(self.cards["Measured"])
        config_layout.addWidget(self.cards["Indicated"])
        config_layout.addWidget(self.cards["Inferred"])
        
        config_layout.addStretch()
        scroll.setWidget(config_widget)
        splitter.addWidget(scroll)
        
        # RIGHT: Preview
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(10, 20, 10, 20)
        
        lbl_prev = QLabel("Classification Results")
        lbl_prev.setStyleSheet("font-size: 12pt; font-weight: bold; color: #4da6ff;")
        preview_layout.addWidget(lbl_prev)
        
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["Category", "Blocks", "%"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("background: #1a1a20; border: none;")
        preview_layout.addWidget(self.table)
        
        self.btn_run = QPushButton("RUN CLASSIFICATION")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; padding: 15px; font-weight: bold; font-size: 11pt;")
        self.btn_run.clicked.connect(self.run_classification)
        preview_layout.addWidget(self.btn_run)
        
        # Visualization Button
        self.btn_viz = QPushButton("Visualize in 3D")
        self.btn_viz.setStyleSheet("background-color: #1976d2; color: white; padding: 15px; font-weight: bold; font-size: 11pt;")
        self.btn_viz.clicked.connect(self._visualize_results)
        self.btn_viz.setEnabled(False)
        preview_layout.addWidget(self.btn_viz)
        
        splitter.addWidget(preview_widget)
        # FIXED: 50/50 split - results are important!
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([750, 750])  # Equal initial sizes

        self.main_layout.addWidget(splitter)
        
        self._on_var_changed() # Init distances

    def _apply_theme(self):
        """Apply modern high-contrast stylesheet for better visibility."""
        colors = get_theme_colors()
        base_style = get_analysis_panel_stylesheet()

        # Add panel-specific enhancements
        panel_specific = f"""
            /* High-contrast labels */
            QLabel {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: 10pt;
            }}

            /* Professional GroupBox styling */
            QGroupBox {{
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 6px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #222222;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                color: #3498db;
                padding: 0 5px;
            }}

            /* Enhanced inputs */
            QDoubleSpinBox, QSpinBox, QComboBox {{
                background-color: #111;
                border: 1px solid #555;
                color: white;
                padding: 4px;
                min-height: 25px;
            }}

            /* Table styling */
            QTableWidget {{
                gridline-color: #444;
                background-color: {{colors.CARD_BG}};
                color: {{colors.TEXT_PRIMARY}};
            }}

            QHeaderView::section {{
                background-color: #2a2a2a;
                color: {{colors.TEXT_PRIMARY}};
                padding: 6px;
                border: 1px solid #444;
                font-weight: bold;
            }}
        """

        self.setStyleSheet(base_style + panel_specific)

    def _create_header(self):
        frame = QFrame()
        frame.setFixedHeight(50)
        frame.setStyleSheet("background: #151518; border-bottom: 1px solid #303038;")
        layout = QHBoxLayout(frame)
        
        self.status_lbl = QLabel("WAITING FOR DATA")
        self.status_lbl.setStyleSheet("color: #777; font-weight: bold;")
        
        self.domain_combo = QComboBox()
        self.domain_combo.addItem("Full Model Extent")
        
        layout.addWidget(QLabel("Resource Classification"))
        layout.addStretch()
        
        # Refresh button to reload data from registry
        self.refresh_btn = QPushButton("⟳ Refresh")
        self.refresh_btn.setToolTip("Reload data from registry (Block Model, Drillholes)")
        self.refresh_btn.setStyleSheet("""
            QPushButton { 
                background: #303038; color: #4da6ff; border: 1px solid #4da6ff; 
                padding: 4px 12px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background: #404050; }
        """)
        self.refresh_btn.clicked.connect(self._refresh_data)
        layout.addWidget(self.refresh_btn)
        
        layout.addSpacing(10)
        layout.addWidget(QLabel("Domain:"))
        layout.addWidget(self.domain_combo)
        layout.addSpacing(20)
        layout.addWidget(self.status_lbl)
        return frame

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

                # Multi-model support - current model changed
                safe_connect('currentBlockModelChanged', self._on_bm_loaded)
                
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
                logger.info(f"Resource Classification: Connected {len(connected_signals)} signals: {connected_signals}")
        except Exception as e:
            logger.warning(f"Resource Classification: Failed to connect to registry: {e}", exc_info=True)

    def _on_estimation_results(self, results):
        """Handle estimation/simulation results as block model source."""
        logger.info("Resource Classification: Received estimation results")

        if results is None:
            return

        # Check if this is SGSIM results (has realizations or multiple statistics)
        if isinstance(results, dict):
            if 'realizations' in results or any('SGSIM' in str(k).upper() for k in results.keys()):
                # Register SGSIM sources separately
                self._register_sgsim_sources(results)
                self._update_source_selector()
                return

        # For other estimation methods (Kriging, etc.)
        df = self._extract_block_model_from_results(results)
        if df is not None and not df.empty:
            # Try to determine the estimation method
            method = "Estimation"
            if isinstance(results, dict):
                method = results.get('method', results.get('estimation_method', 'Kriging'))

            source_name = f"{method} Results"
            self._register_source(source_name, df, auto_select=True)
            self._update_source_selector()
            logger.info(f"Resource Classification: Registered {source_name} ({len(df)} blocks)")
    
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
                        logger.info(f"Resource Classification: Extracted {len(df):,} blocks from estimation grid arrays")
                        return ensure_xyz_columns(df)
                except Exception as e:
                    logger.warning(f"Resource Classification: Failed to extract grid arrays: {e}")
            
            # Priority 2: Check for DataFrame keys
            for key in ['data', 'df', 'block_model', 'results', 'grid', 'classified_df']:
                if key in results:
                    candidate = results[key]
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        logger.info(f"Resource Classification: Extracted {len(candidate):,} blocks from '{key}' key")
                        return ensure_xyz_columns(candidate)
            
            # Priority 3: Try the dict itself if it has X, Y, Z columns
            if all(k in results for k in ['X', 'Y', 'Z']):
                df = pd.DataFrame(results)
                if not df.empty:
                    return ensure_xyz_columns(df)
                    
        elif isinstance(results, pd.DataFrame):
            return ensure_xyz_columns(results)
        
        return None

    def _refresh_data(self):
        """Manual refresh button handler - reload all data from registry."""
        logger.info("Resource Classification: Manual refresh requested")

        # Clear existing sources and reload
        self._block_model_sources.clear()
        self._available_sources.clear()
        self._current_source = ""

        self._load_existing()

        # Show feedback to user
        source_count = len(self._available_sources)
        if self.drillhole_data is not None and source_count > 0:
            source_list = "\n".join([f"  • {s}" for s in self._available_sources[:5]])
            if source_count > 5:
                source_list += f"\n  ... and {source_count - 5} more"
            QMessageBox.information(self, "Refresh Complete",
                f"Data refreshed successfully!\n\n"
                f"• Drillholes: {len(self.drillhole_data):,} samples\n"
                f"• Block Model Sources: {source_count}\n{source_list}")
        elif self.drillhole_data is None and source_count == 0:
            QMessageBox.warning(self, "No Data Found",
                "No drillhole or block model data found in registry.\n\n"
                "Please ensure you have:\n"
                "1. Loaded drillhole data\n"
                "2. Built or loaded a block model, OR\n"
                "3. Run SGSIM simulation")
        else:
            missing = []
            if self.drillhole_data is None:
                missing.append("• Drillhole data (run compositing or load assays)")
            if self.block_model_data is None:
                missing.append("• Block model (use Block Model Builder, run Kriging/SGSIM, or load from file)")
            QMessageBox.warning(self, "Missing Data", 
                f"Some data is still missing:\n\n" + "\n".join(missing))

    def _load_existing(self):
        """Load existing data from registry on panel open."""
        if not hasattr(self, 'reg') or self.reg is None:
            logger.warning("Resource Classification: Registry not available")
            return

        logger.info("Resource Classification: Loading existing data from registry...")

        # 1. Load drillhole data (prioritize composites)
        try:
            dh = self.reg.get_drillhole_data()
            if dh is not None:
                if isinstance(dh, dict):
                    comp = dh.get('composites')
                    assays = dh.get('assays')
                    comp_count = len(comp) if isinstance(comp, pd.DataFrame) and not comp.empty else 0
                    assay_count = len(assays) if isinstance(assays, pd.DataFrame) and not assays.empty else 0
                    logger.info(f"Resource Classification: Found drillhole data - {comp_count} composites, {assay_count} assays")
                self._on_dh_loaded(dh)
            else:
                logger.info("Resource Classification: No drillhole data in registry")
        except Exception as e:
            logger.warning(f"Resource Classification: Failed to load drillhole data: {e}")

        # 2. Load all registered block models (multi-model support)
        try:
            models = self.reg.get_block_model_list()
            if models:
                logger.info(f"Resource Classification: Found {len(models)} registered block model(s)")
                for i, model_info in enumerate(models):
                    model_id = model_info['model_id']
                    row_count = model_info['row_count']
                    is_current = model_info['is_current']

                    # Retrieve the actual model data
                    bm = self.reg.get_block_model(model_id=model_id)
                    if bm is not None:
                        df = self._convert_to_dataframe(bm)
                        if df is not None and not df.empty:
                            # Use model_id as source name
                            display_name = f"{model_id}{' (current)' if is_current else ''}"
                            # Auto-select the current model
                            self._register_source(display_name, df, auto_select=is_current)
                            logger.info(f"Resource Classification: Loaded '{model_id}' ({row_count:,} blocks)")
        except Exception as e:
            logger.warning(f"Resource Classification: Failed to load block models: {e}")

        # 3. Load classified block model if available (legacy support)
        try:
            classified = self.reg.get_classified_block_model()
            if classified is not None:
                df = self._convert_to_dataframe(classified)
                if df is not None and not df.empty:
                    # Check if this is already registered as a multi-model
                    already_registered = any('classified' in name.lower() for name in self._block_model_sources.keys())
                    if not already_registered:
                        self._register_source("Block Model (Classified)", df, auto_select=False)
                        logger.info(f"Resource Classification: Found classified block model ({len(df):,} blocks)")
        except Exception as e:
            logger.debug(f"Resource Classification: No classified block model: {e}")

        # 4. Load SGSIM results - register individual statistics
        try:
            if hasattr(self.reg, 'get_sgsim_results'):
                sgsim = self.reg.get_sgsim_results()
                if sgsim is not None:
                    self._register_sgsim_sources(sgsim)
        except Exception as e:
            logger.debug(f"Resource Classification: No SGSIM results: {e}")

        # 5. Load other estimation results (Kriging etc.)
        self._try_load_estimation_results()

        # 6. Update source selector UI
        self._update_source_selector()

        # 7. Log final status
        dh_status = "✓" if self.drillhole_data is not None else "✗"
        bm_status = "✓" if self.block_model_data is not None else "✗"
        logger.info(f"Resource Classification: Load complete - Drillholes: {dh_status}, Block Model: {bm_status}")
    
    def _try_load_estimation_results(self):
        """Try to load block model data from ALL estimation/simulation results."""
        if not hasattr(self, 'reg') or self.reg is None:
            return

        # List of estimation result getters to try (register ALL available)
        estimation_sources = [
            ('kriging_results', 'get_kriging_results', 'Ordinary Kriging'),
            ('simple_kriging_results', 'get_simple_kriging_results', 'Simple Kriging'),
            ('universal_kriging_results', 'get_universal_kriging_results', 'Universal Kriging'),
            ('cokriging_results', 'get_cokriging_results', 'Co-Kriging'),
            ('indicator_kriging_results', 'get_indicator_kriging_results', 'Indicator Kriging'),
            ('soft_kriging_results', 'get_soft_kriging_results', 'Soft Kriging'),
        ]

        found_any = False
        for key, getter_name, source_name in estimation_sources:
            if hasattr(self.reg, getter_name):
                try:
                    results = getattr(self.reg, getter_name)()
                    if results is not None:
                        df = self._extract_block_model_from_results(results)
                        if df is not None and not df.empty:
                            # Register as a selectable source
                            self._register_source(f"{source_name} Results", df, auto_select=not found_any)
                            found_any = True
                            logger.info(f"Resource Classification: Registered {source_name} ({len(df)} blocks)")
                except Exception as e:
                    logger.debug(f"Resource Classification: Failed to load {source_name}: {e}")

        if not found_any:
            logger.info("Resource Classification: No estimation results found in registry")

    def _on_dh_loaded(self, data):
        """Load drillhole data, PRIORITIZING COMPOSITES over raw assays.
        
        Data sources:
        - Compositing Window: Saves composites to registry via drillholeDataLoaded signal
        - Variogram Panel: Uses composites from registry for analysis
        - Drillhole Import: Provides raw assay data
        
        LINEAGE: This method tracks data source type for audit compliance.
        """
        # Log diagnostic info about registry contents
        log_registry_data_status("Resource Classification", data)
        
        df = None
        source_type = "unknown"
        
        if isinstance(data, dict):
            composites = data.get('composites')
            assays = data.get('assays')
            
            # Priority 1: Composites (from Compositing Window or other sources)
            if isinstance(composites, pd.DataFrame) and not composites.empty:
                df = composites
                source_type = "composites"
                logger.info(f"Resource Classification: Using COMPOSITES data ({len(df)} samples)")
            # Priority 2: Raw assays as fallback (WITH LINEAGE WARNING)
            elif isinstance(assays, pd.DataFrame) and not assays.empty:
                df = assays
                source_type = "assays"
                logger.warning(
                    f"LINEAGE WARNING: Resource Classification using RAW ASSAYS ({len(df)} samples). "
                    "JORC/SAMREC Best Practice: Use composited data for defensible classification. "
                    "Run compositing first for compliant resource estimation."
                )
                # Store lineage flag for downstream audit
                self._data_source_is_raw_assays = True
            else:
                logger.warning("Resource Classification: Drillhole dict received but no valid composites or assays found")
        elif isinstance(data, pd.DataFrame) and not data.empty:
            df = data
            source_type = "dataframe"
            logger.info(f"Resource Classification: Using DataFrame ({len(df)} samples)")
        else:
            logger.warning(f"Resource Classification: Invalid drillhole data type: {type(data)}")
        
        # Track data source for lineage
        self._drillhole_source_type = source_type
        
        # Ensure proper coordinate columns
        if df is not None:
            self.drillhole_data = ensure_xyz_columns(df)
            logger.info(f"Resource Classification: Loaded {len(self.drillhole_data)} drillhole samples ({source_type})")
        else:
            self.drillhole_data = None
        
        self._check_ready()

    def _on_bm_loaded(self, bm):
        """Handle block model data (from file, builder, or estimation results).
        
        Data sources:
        - Block Model Loading: CSV/Parquet block model files
        - Block Model Builder: Generated from drillhole data + grid definition
        - Kriging/SGSIM: Estimation results as block model
        """
        try:
            if bm is None:
                logger.warning("Resource Classification: Received None block model")
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
                logger.warning(f"Resource Classification: Unknown block model type: {type(bm)}")
                return
            
            if df is None or df.empty:
                logger.warning("Resource Classification: Block model DataFrame is empty")
                return
            
            self.block_model_data = ensure_xyz_columns(df)
            logger.info(f"Resource Classification: Loaded block model ({source_type}) with {len(self.block_model_data):,} blocks")
        except Exception as e:
            logger.error(f"Resource Classification: Error loading block model: {e}", exc_info=True)
            return
        
        self._check_ready()
        
        # Update domains
        self.domain_combo.clear()
        self.domain_combo.addItem("Full Model Extent")
        for col in ['DOMAIN', 'Domain', 'ZONE', 'Zone', 'domain', 'zone']:
            if col in self.block_model_data.columns:
                unique_domains = self.block_model_data[col].dropna().unique()
                for d in sorted(unique_domains):
                    self.domain_combo.addItem(str(d))
                logger.info(f"Resource Classification: Found {len(unique_domains)} domains in column '{col}'")
                break

    def _convert_to_dataframe(self, data) -> Optional[pd.DataFrame]:
        """Convert various data types to DataFrame."""
        if data is None:
            return None
        if isinstance(data, pd.DataFrame):
            return data
        if hasattr(data, 'to_dataframe'):
            return data.to_dataframe()
        if hasattr(data, 'data') and isinstance(data.data, pd.DataFrame):
            return data.data
        if isinstance(data, dict):
            return self._extract_block_model_from_results(data)
        return None

    def _register_source(self, name: str, data: pd.DataFrame, auto_select: bool = False):
        """Register a block model source."""
        if data is None or data.empty:
            return

        # Ensure XYZ columns
        df = ensure_xyz_columns(data)
        if df is None or df.empty:
            return

        self._block_model_sources[name] = df
        if name not in self._available_sources:
            self._available_sources.append(name)

        logger.info(f"Resource Classification: Registered source '{name}' ({len(df):,} blocks)")

        # Auto-select if requested and no current selection
        if auto_select and (not self._current_source or self._current_source == ""):
            self._current_source = name
            self.block_model_data = df
            self._check_ready()

    def _register_sgsim_sources(self, sgsim_results: Dict[str, Any]):
        """Register SGSIM results as multiple block model sources."""
        import pyvista as pv

        if sgsim_results is None:
            return

        variable = sgsim_results.get('variable', 'Grade')

        # Check for PyVista grid with multiple properties
        grid = sgsim_results.get('grid') or sgsim_results.get('pyvista_grid')
        if grid is not None and isinstance(grid, (pv.RectilinearGrid, pv.UnstructuredGrid, pv.StructuredGrid, pv.ImageData)):
            if hasattr(grid, 'cell_centers'):
                centers = grid.cell_centers()
                if hasattr(centers, 'points'):
                    coords = centers.points
                    base_df = pd.DataFrame({
                        'X': coords[:, 0],
                        'Y': coords[:, 1],
                        'Z': coords[:, 2]
                    })

                    # Register each property as a separate source
                    for prop_name in grid.cell_data.keys():
                        df = base_df.copy()
                        df[prop_name] = grid.cell_data[prop_name]

                        # Determine display name
                        if 'MEAN' in prop_name.upper():
                            display_name = f"SGSIM Mean ({variable})"
                        elif 'P10' in prop_name.upper():
                            display_name = f"SGSIM P10 ({variable})"
                        elif 'P50' in prop_name.upper():
                            display_name = f"SGSIM P50 ({variable})"
                        elif 'P90' in prop_name.upper():
                            display_name = f"SGSIM P90 ({variable})"
                        elif 'STD' in prop_name.upper():
                            display_name = f"SGSIM Std Dev ({variable})"
                        elif 'PROB' in prop_name.upper():
                            display_name = f"SGSIM Probability ({prop_name})"
                        else:
                            display_name = f"SGSIM {prop_name}"

                        self._register_source(display_name, df, auto_select=False)

        # Also check for realizations array (compute E-type)
        if 'realizations' in sgsim_results:
            reals = sgsim_results['realizations']
            grid_x = sgsim_results.get('grid_x')
            grid_y = sgsim_results.get('grid_y')
            grid_z = sgsim_results.get('grid_z')

            if grid_x is not None and isinstance(reals, np.ndarray):
                # E-type (mean of all realizations)
                mean_estimate = np.mean(reals, axis=0) if reals.ndim == 2 else reals.ravel()
                df = pd.DataFrame({
                    'X': np.asarray(grid_x).ravel(),
                    'Y': np.asarray(grid_y).ravel(),
                    'Z': np.asarray(grid_z).ravel(),
                    variable: mean_estimate
                })
                self._register_source(f"SGSIM E-type Mean ({variable})", df, auto_select=False)

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

            # Select current source
            if self._current_source and self._current_source in self._available_sources:
                idx = self._available_sources.index(self._current_source)
                self.source_combo.setCurrentIndex(idx)
                df = self._block_model_sources.get(self._current_source)
                if df is not None:
                    cols = [c for c in df.columns if c.upper() not in ('X', 'Y', 'Z')]
                    self.source_info_label.setText(f"Properties: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")

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
            self._check_ready()
            return

        if source_name in self._block_model_sources:
            self._current_source = source_name
            self.block_model_data = self._block_model_sources[source_name]

            # Update info label
            df = self.block_model_data
            cols = [c for c in df.columns if c.upper() not in ('X', 'Y', 'Z')]
            self.source_info_label.setText(
                f"Selected: {len(df):,} blocks | Properties: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}"
            )

            logger.info(f"Resource Classification: Switched to '{source_name}' ({len(df):,} blocks)")

            # Update domains
            self.domain_combo.clear()
            self.domain_combo.addItem("Full Model Extent")
            for col in ['DOMAIN', 'Domain', 'ZONE', 'Zone', 'domain', 'zone']:
                if col in df.columns:
                    unique_domains = df[col].dropna().unique()
                    for d in sorted(unique_domains):
                        self.domain_combo.addItem(str(d))
                    break

            self._check_ready()

    def _check_ready(self):
        """Check if all required data is available and update UI status."""
        has_drillholes = self.drillhole_data is not None
        has_block_model = self.block_model_data is not None
        ready = has_drillholes and has_block_model
        
        if ready:
            dh_count = len(self.drillhole_data)
            bm_count = len(self.block_model_data)
            self.status_lbl.setText(f"READY ({dh_count:,} samples, {bm_count:,} blocks)")
            self.status_lbl.setStyleSheet("color: #4caf50; font-weight: bold; background: #202020; padding: 5px 10px; border-radius: 4px;")
            logger.info(f"Resource Classification: READY - {dh_count} drillhole samples, {bm_count} blocks")
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
            logger.info(f"Resource Classification: Not ready - {status_text}")

    def _on_var_changed(self):
        r = self.spin_range.value()
        for card in self.cards.values():
            card.set_variogram_range(r)

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

    def run_classification(self):
        """Run classification in background thread to prevent UI freeze."""
        if self.drillhole_data is None or self.block_model_data is None:
            QMessageBox.warning(self, "Error", "Missing data.")
            return

        # LINEAGE CHECK: Warn if data source was raw assays
        if getattr(self, '_data_source_is_raw_assays', False):
            reply = QMessageBox.warning(
                self,
                "Lineage Warning: Raw Assays",
                "The current drillhole data is from RAW ASSAYS, not composites.\n\n"
                "JORC/SAMREC Best Practice:\n"
                "Resource classification should be based on estimation results\n"
                "derived from composited data. Using raw assays may result in:\n"
                "• Inconsistent sample support\n"
                "• Unreliable classification boundaries\n"
                "• Non-defensible resource statements\n\n"
                "Do you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            logger.warning("LINEAGE: User proceeding with classification using raw assay-derived data")

        # UI State
        self.btn_run.setText("RUNNING... (PLEASE WAIT)")
        self.btn_run.setEnabled(False)
        self.btn_viz.setEnabled(False)
        # Force UI update using safer processEvents
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            # Setup Params
            var = VariogramModel(range_major=self.spin_range.value(), sill=self.spin_sill.value())
            
            p_meas = self.cards["Measured"].get_params()
            p_ind = self.cards["Indicated"].get_params()
            p_inf = self.cards["Inferred"].get_params()
            
            dom = self.domain_combo.currentText()
            dom_val = dom if dom != "Full Model Extent" else None
            
            rules = ClassificationRuleset.from_ui_params(
                meas_dist_pct=p_meas['dist_pct'], meas_min_holes=p_meas['min_holes'],
                ind_dist_pct=p_ind['dist_pct'], ind_min_holes=p_ind['min_holes'],
                inf_dist_pct=p_inf['dist_pct'], inf_min_holes=p_inf['min_holes'],
                domain_name=dom_val
            )
            
            # Audit Logging: Log the classification start with parameters
            from ..core.audit_manager import AuditManager
            audit = AuditManager()
            
            # Hash input data for audit trail
            input_hash_bm = audit.hash_dataframe(self.block_model_data)
            input_hash_dh = audit.hash_dataframe(self.drillhole_data) if self.drillhole_data is not None else "N/A"
            
            # Log classification parameters
            params_dict = {
                "variogram": {"range_major": var.range_major, "sill": var.sill},
                "rules": rules.to_dict(),
                "domain": dom_val or "Full Model Extent",
                "input_hash_block_model": input_hash_bm,
                "input_hash_drillholes": input_hash_dh
            }
            
            event_id = audit.log_event(
                module="ResourceClass",
                action="Classify",
                parameters=params_dict,
                input_data_hash=f"{input_hash_bm}_{input_hash_dh}"
            )
            
            # Store event_id for later use in completion handler
            self._last_audit_event_id = event_id
            
            if not self.controller:
                QMessageBox.warning(self, "Error", "Controller not available.")
                self.btn_run.setText("RUN CLASSIFICATION")
                self.btn_run.setEnabled(True)
                return

            # ================================================================
            # CRITICAL FIX: Align coordinate systems before classification
            # ================================================================
            # Prevents coordinate mismatch when block model is in UTM (~500,000m)
            # and drillholes are in local coordinates (~0m), or vice versa.
            # Without this, distance calculations will be meaningless.
            # ================================================================
            aligned_bm, aligned_dh = self._align_coordinate_systems(
                self.block_model_data.copy(),
                self.drillhole_data.copy() if self.drillhole_data is not None else None
            )

            # Prepare params for task
            task_params = {
                "block_model": aligned_bm,
                "drillhole_df": aligned_dh,
                "jorc_variogram": {"range_major": var.range_major, "sill": var.sill},
                "jorc_rules": rules.to_dict(),
                "domain_value": dom_val
            }
            
            # Progress callback - update UI during classification
            def progress_callback(percent: int, message: str):
                if hasattr(self, 'progress_bar') and self.progress_bar:
                    self.progress_bar.setValue(percent)
                # Force UI repaint using safer processEvents
                from PyQt6.QtCore import QEventLoop
                from PyQt6.QtWidgets import QApplication
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            
            # Run via controller
            self.controller.run_task(
                'classify',
                task_params,
                callback=self._on_classification_complete,
                progress_callback=progress_callback
            )
            
        except Exception as e:
            self.btn_run.setText("RUN CLASSIFICATION")
            self.btn_run.setEnabled(True)
            QMessageBox.critical(self, "Setup Error", str(e))
    
    def _on_classification_complete(self, result: Dict[str, Any]):
        """Handle successful classification completion."""
        if result is None:
            self._on_worker_error("Classification returned no result.")
            return
        
        if result.get("error"):
            self._on_worker_error(result["error"])
            return
        
        # --- NEW: Handle BlockModel API (standard) ---
        from ..models.block_model import BlockModel
        
        # Check if BlockModel was returned with classification results
        block_model = result.get("block_model")
        if isinstance(block_model, BlockModel):
            # ✅ STANDARD API: Classification results already added to BlockModel
            logger.info("✅ BlockModel API: Classification results added to BlockModel")
            
            # Extract ClassificationResult from BlockModel properties
            # The engine should have added 'Category' or 'CLASS_FINAL' property
            category_prop = block_model.get_property('Category') or block_model.get_property('CLASS_FINAL')
            
            if category_prop is not None:
                # Create ClassificationResult for UI compatibility
                from ..models.jorc_classification_engine import ClassificationResult
                
                # Convert BlockModel to DataFrame for UI display
                classified_df = block_model.to_dataframe()
                
                # Create result object
                res = ClassificationResult(
                    classified_df=classified_df,
                    summary=result.get("summary", {}),
                    ruleset=result.get("ruleset"),
                    variogram=result.get("variogram"),
                    domain_name=result.get("domain_name"),
                    audit_records=result.get("audit_records", []),
                    execution_time_seconds=result.get("execution_time_seconds", 0.0)
                )
                
                # Register updated BlockModel
                try:
                    if hasattr(self, 'registry') and self.registry:
                        self.registry.register_block_model(block_model, source_panel="ResourceClassificationPanel")
                        logger.info("✅ Registered updated BlockModel with classification results")
                except Exception as e:
                    logger.warning(f"Failed to register BlockModel: {e}")
                
                # Continue with existing UI logic
                self._on_worker_finished(res)
                return
        
        # --- LEGACY: Handle DataFrame (backward compatibility) ---
        res = result.get("classified_df")
        if res is None:
            self._on_worker_error("No classified DataFrame in result.")
            return
        
        # Try to add to existing BlockModel if available
        try:
            if hasattr(self, 'registry') and self.registry:
                existing_bm = self.registry.get_block_model()
                if isinstance(existing_bm, BlockModel) and hasattr(res, 'classified_df'):
                    # Add classification property to existing BlockModel
                    category_col = None
                    for col in ['Category', 'CLASS_FINAL', 'CLASS_AUTO']:
                        if col in res.classified_df.columns:
                            category_col = col
                            break
                    
                    if category_col:
                        # Match by coordinates and add category
                        # This is simplified - in production, match by block ID
                        categories = res.classified_df[category_col].values
                        if len(categories) == existing_bm.block_count:
                            existing_bm.add_property('Category', categories)
                            self.registry.register_block_model(existing_bm, source_panel="ResourceClassificationPanel")
                            logger.info("✅ Added classification results to existing BlockModel")
        except Exception as e:
            logger.warning(f"Failed to add classification to BlockModel: {e}")
        
        # Continue with existing logic
        self._on_worker_finished(res)
    
    def _on_worker_finished(self, res):
        """Handle successful classification completion (internal)."""
        self.classification_result = res
        
        # Audit Logging: Log the classification results
        from ..core.audit_manager import AuditManager
        audit = AuditManager()
        
        # Hash output data for audit trail
        output_hash = audit.hash_dataframe(res.classified_df)
        
        # Log completion with results summary
        result_summary = {
            "output_hash": output_hash,
            "summary": res.summary,
            "execution_time_seconds": getattr(res, 'execution_time_seconds', 0.0),
            "total_blocks": len(res.classified_df)
        }
        
        # Update the original audit log entry with results
        audit.log_event(
            module="ResourceClass",
            action="Classify_Complete",
            parameters={"original_event_id": getattr(self, '_last_audit_event_id', 'N/A')},
            result_summary=result_summary
        )
        
        # Update Table
        summary = res.summary
        for i, cat in enumerate(CLASSIFICATION_ORDER):
            c_data = summary.get(cat, {})
            self.table.setItem(i, 0, QTableWidgetItem(cat))
            self.table.setItem(i, 1, QTableWidgetItem(str(c_data.get('count', 0))))
            self.table.setItem(i, 2, QTableWidgetItem(f"{c_data.get('percentage', 0):.1f}%"))
            
            # Color code
            col = QColor(CLASSIFICATION_COLORS.get(cat, f"{ModernColors.TEXT_PRIMARY}"))
            self.table.item(i, 0).setForeground(col)

        self.classification_complete.emit(res)
        
        # Restore UI
        self.btn_run.setText("RUN CLASSIFICATION")
        self.btn_run.setEnabled(True)
        self.btn_viz.setEnabled(True)
        QMessageBox.information(self, "Success", "Classification Complete")
    
    def _on_worker_error(self, msg):
        """Handle classification error."""
        self.btn_run.setText("RUN CLASSIFICATION")
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Calculation Error", msg)

    def _visualize_results(self):
        """Visualize classification results in 3D as blocks (optimized for large models)."""
        if self.classification_result is None:
            logger.warning("Cannot visualize: classification_result is None")
            QMessageBox.warning(self, "No Results", "No classification results to visualize. Run classification first.")
            return

        try:
            import pyvista as pv

            logger.info("Starting classification visualization...")
            df = self.classification_result.classified_df
            logger.info(f"Classification DataFrame has {len(df)} blocks")
            logger.info(f"DataFrame columns: {list(df.columns)}")

            # Robust coordinate check
            coords = None
            coord_names = None
            for c in [['XC', 'YC', 'ZC'], ['X', 'Y', 'Z'], ['x', 'y', 'z']]:
                if all(col in df.columns for col in c):
                    coords = df[c].values
                    coord_names = c
                    break

            if coords is None:
                error_msg = f"Could not find coordinate columns. Available columns: {list(df.columns)}"
                logger.error(error_msg)
                QMessageBox.warning(self, "Error", f"Could not find coordinate columns (XC/YC/ZC or X/Y/Z)\n\nAvailable columns: {list(df.columns)}")
                return

            logger.info(f"Using coordinate columns: {coord_names}")
            logger.info(f"Coordinate bounds: X=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
                       f"Y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
                       f"Z=[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

            # Check for CLASS_FINAL column
            if "CLASS_FINAL" not in df.columns:
                error_msg = f"CLASS_FINAL column not found. Available columns: {list(df.columns)}"
                logger.error(error_msg)
                QMessageBox.warning(self, "Error", f"CLASS_FINAL column not found in classification results.\n\nAvailable columns: {list(df.columns)}")
                return

            # Map classes to numeric codes for coloring
            # Measured=3, Indicated=2, Inferred=1, Unclassified=0
            classes = df["CLASS_FINAL"].values
            mapping = {"Measured": 3, "Indicated": 2, "Inferred": 1, "Unclassified": 0}
            scalars = np.array([mapping.get(c, 0) for c in classes])

            # Log classification distribution
            unique, counts = np.unique(classes, return_counts=True)
            logger.info(f"Classification distribution: {dict(zip(unique, counts))}")

            # Create PolyData
            pdata = pv.PolyData(coords)
            pdata["Classification"] = scalars
            pdata.set_active_scalars("Classification")  # CRITICAL: Set active scalars
            logger.info(f"Created PolyData with {pdata.n_points} points")
            logger.info(f"PolyData point_data arrays: {list(pdata.point_data.keys())}")

            # Store original categorical values for custom color definition
            # This allows users to define colors for "Measured", "Indicated", etc.
            pdata["Classification_Categories"] = classes

            # Performance Optimization:
            # If blocks > 500k, use Point Gaussian (splats) instead of Cubes
            # It looks almost the same but renders 10x faster.

            if len(df) > 500000:
                # Point Cloud Representation (much faster for huge models)
                # The viewer should handle rendering with render_points_as_spheres=True
                logger.info("Using point cloud representation (>500k blocks)")
                logger.info(f"Emitting visualization signal with {pdata.n_points} points")
                self.request_visualization.emit(pdata, "Classification")
            else:
                # Cube Glyph Representation (Prettier for small/medium models)
                dx = df["DX"].values[0] if "DX" in df.columns else 10.0
                dy = df["DY"].values[0] if "DY" in df.columns else 10.0
                dz = df["DZ"].values[0] if "DZ" in df.columns else 5.0

                logger.info(f"Creating cube glyphs with dimensions: dx={dx}, dy={dy}, dz={dz}")

                cube = pv.Box(bounds=(-dx/2, dx/2, -dy/2, dy/2, -dz/2, dz/2))

                # CRITICAL FIX: Glyph with active scalars set
                # This transfers point data to cell data in the glyphs
                glyphs = pdata.glyph(geom=cube, scale=False, orient=False)

                # Ensure scalars are in cell_data for the glyphs (glyph converts point->cell)
                # The glyph() method should automatically transfer active scalars from point to cell data
                # But we verify and set active scalars explicitly
                if "Classification" not in glyphs.cell_data and "Classification" in pdata.point_data:
                    # Manual transfer if automatic failed
                    glyphs.cell_data["Classification"] = pdata.point_data["Classification"]
                    logger.info("Manually transferred Classification from point_data to cell_data")

                if "Classification" in glyphs.cell_data:
                    glyphs.set_active_scalars("Classification")
                    logger.info("Set Classification as active scalars on glyphs (cell_data)")
                else:
                    logger.error("Classification data not found in glyphs cell_data!")

                logger.info(f"Created glyphs with {glyphs.n_cells} cells")
                logger.info(f"Glyphs cell_data arrays: {list(glyphs.cell_data.keys())}")
                logger.info(f"Glyphs active scalars: {glyphs.active_scalars_name}")
                logger.info(f"Emitting visualization signal with {glyphs.n_cells} block glyphs")
                self.request_visualization.emit(glyphs, "Classification")

            logger.info("Classification visualization signal emitted successfully")

        except Exception as e:
            logger.exception("Visualization error")
            QMessageBox.warning(self, "Visualization Error",
                              f"Failed to visualize classification results:\n\n{str(e)}\n\n"
                              f"Check the console log for full details.")

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Variogram parameters
            settings['range_major'] = get_safe_widget_value(self, 'spin_range_major')
            settings['range_minor'] = get_safe_widget_value(self, 'spin_range_minor')
            settings['range_vert'] = get_safe_widget_value(self, 'spin_range_vert')
            settings['sill'] = get_safe_widget_value(self, 'spin_sill')
            settings['nugget'] = get_safe_widget_value(self, 'spin_nugget')
            
            # Domain
            settings['domain'] = get_safe_widget_value(self, 'domain_combo')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save resource classification panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Variogram parameters
            set_safe_widget_value(self, 'spin_range_major', settings.get('range_major'))
            set_safe_widget_value(self, 'spin_range_minor', settings.get('range_minor'))
            set_safe_widget_value(self, 'spin_range_vert', settings.get('range_vert'))
            set_safe_widget_value(self, 'spin_sill', settings.get('sill'))
            set_safe_widget_value(self, 'spin_nugget', settings.get('nugget'))
            
            # Domain
            set_safe_widget_value(self, 'domain_combo', settings.get('domain'))
                
            logger.info("Restored resource classification panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore resource classification panel settings: {e}")

# Alias for backward compatibility
ResourceClassificationPanel = JORCClassificationPanel
