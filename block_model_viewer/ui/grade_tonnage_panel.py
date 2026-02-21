"""
Grade-Tonnage Analysis and Cut-off Sensitivity Panel.
Modern UX/UI with dual-mode support for block models and composites.

Features:
- Dual mode: BLOCK_MODEL (no declustering) vs COMPOSITES (declustered)
- Proper tonnage anchoring to deposit totals
- Cut-off sensitivity analysis with NPV optimization
- Heuristic uncertainty bands (±CV, not formal CIs)
- Economic parameter sensitivity analysis
- Numba JIT optimization for performance

IMPORTANT NOTES:
- For kriged block models, use BLOCK_MODEL mode (no declustering)
- For drillhole composites, use COMPOSITES mode (cell declustering)
- Uncertainty bands are heuristic ±CV, not formal geostatistical CIs
- For proper uncertainty, integrate with SGS realisations

Author: GeoX Mining Software - Grade-Tonnage Panel
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QRadioButton,
    QCheckBox, QSplitter, QTabWidget, QHeaderView, QFormLayout, QFrame, QScrollArea,
    QProgressBar, QTextEdit, QSpinBox, QCheckBox, QButtonGroup, QDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont

# Matplotlib imports for interactive charts
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    NavigationToolbar = None

# Import geostatistical engines
from ..mine_planning.cutoff.geostats_grade_tonnage import (
    GeostatsGradeTonnageEngine,
    CutoffSensitivityEngine,
    GeostatsGradeTonnageConfig,
    CutoffOptimizationMethod,
    ConfidenceIntervalMethod,
    DataMode,
    GradeWeightingMethod,
    validate_grade_tonnage_data
)

# Import multi-period mine economics
from ..mine_planning.cutoff.mine_economics import (
    MineEconomicsEngine,
    MineEconomicsConfig,
    EconomicParameters,
    MineCapacity,
    CapitalExpenditure,
    TaxParameters,
    SensitivityAnalyzer
)

# Import advanced analysis engines (JORC/SAMREC compliance)
from ..mine_planning.cutoff.gt_advanced_analysis import (
    SGSUncertaintyEngine,
    DomainGTEngine,
    ClassificationGTEngine,
    SGSUncertaintyResult,
    DomainGTResult,
    ClassificationGTResult,
    export_sgs_uncertainty_to_csv,
    export_domain_gt_to_csv,
    export_classification_gt_to_csv
)

# Import declustering classes
from ..drillholes.declustering import (
    DeclusteringMethod,
    CellDefinition
)

# Import visualization system
from ..mine_planning.cutoff.advanced_visualization import (
    MiningVisualizationCoordinator,
    PlotConfig,
    PlotType,
    ColorScheme
)

from .base_panel import BasePanel
from .comparison_utils import ComparisonColors, SourceSelectionWidget, create_comparison_legend
from .modern_styles import get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class GradeTonnagePanel(BasePanel):

    # Signals for analysis completion
    analysisCompleted = pyqtSignal(object, object)  # grade_tonnage_curve, sensitivity_analysis
    analysisProgress = pyqtSignal(str)  # progress message
    analysisError = pyqtSignal(str)  # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block_model_data = None
        self.grade_tonnage_curve = None
        self.sensitivity_analysis = None
        self._registry_initialized = False

        # Separate storage for different block model sources
        self._stored_block_model = None
        self._stored_classified_block_model = None
        self._stored_sgsim_results = None
        self._stored_sgsim_df = None
        self._available_sources: list = []
        self._current_source: str = "block_model"

        # Storage for individual SGSIM statistics (Mean, P10, P50, P90, Std Dev)
        self._block_model_sources: Dict[str, Any] = {}

        # Comparison mode state
        self._comparison_mode: bool = False
        self._comparison_results: Dict[str, Any] = {}

        # Initialize engines
        self.gt_engine = GeostatsGradeTonnageEngine()
        self.sensitivity_engine = CutoffSensitivityEngine()
        self.viz_coordinator = MiningVisualizationCoordinator()

        # Analysis state
        self.is_analyzing = False
        self.analysis_thread = None

        # Initialize registry after UI is set up
        # setup_ui() is called by BasePanel.__init__ via _setup_base_ui()
        self._init_registry()

    @property
    def block_model_data(self):
        """Get the block model DataFrame."""
        return self._block_model_data

    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if self.registry:
                logger.info(f"Grade-Tonnage: Successfully connected to registry (controller={self.controller is not None})")
                # Listen to both regular and classified block models with SEPARATE handlers
                self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
                self.registry.blockModelGenerated.connect(self._on_block_model_loaded)
                self.registry.blockModelClassified.connect(self._on_block_model_classified)

                # SGSIM results - direct access without requiring classification
                if hasattr(self.registry, 'sgsimResultsLoaded'):
                    self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)

                # Only check for existing block models if we don't already have data
                # (to avoid overwriting data that was explicitly set)
                if self._block_model_data is None:
                    # Check for regular block model
                    regular_bm = self.registry.get_block_model()
                    if regular_bm:
                        logger.info("Grade-Tonnage: Found regular block model in registry")
                        self._stored_block_model = regular_bm
                        if "block_model" not in self._available_sources:
                            self._available_sources.append("block_model")

                    # Check for classified block model (SEPARATE)
                    classified_bm = self.registry.get_classified_block_model()
                    if classified_bm:
                        logger.info("Grade-Tonnage: Found classified block model in registry")
                        self._stored_classified_block_model = classified_bm
                        if "classified_block_model" not in self._available_sources:
                            self._available_sources.append("classified_block_model")

                    # Check for SGSIM results
                    if hasattr(self.registry, 'get_sgsim_results'):
                        sgsim = self.registry.get_sgsim_results()
                        if sgsim is not None:
                            logger.info("Grade-Tonnage: Found SGSIM results in registry")
                            self._stored_sgsim_results = sgsim
                            # Pre-extract DataFrame for quick access
                            self._on_sgsim_loaded(sgsim)

                    # Update selector and load first available
                    self._update_data_source_selector()

                    # Prefer classified, then regular, then SGSIM
                    if classified_bm:
                        self._current_source = "classified_block_model"
                        self._on_bm_loaded(classified_bm)
                    elif regular_bm:
                        self._current_source = "block_model"
                        self._on_bm_loaded(regular_bm)
                    elif hasattr(self, '_stored_sgsim_df') and self._stored_sgsim_df is not None:
                        self._current_source = "sgsim"
                        self._on_bm_loaded(self._stored_sgsim_df)
                    else:
                        # Also check renderer layers for block models that might be displayed
                        self._check_renderer_layers_for_block_models()
                        if self._block_model_data is None:
                            logger.info("Grade-Tonnage: No block model found in registry yet")
                else:
                    logger.debug("Grade-Tonnage: Already have block model data, skipping registry check")

                self._registry_initialized = True
            else:
                logger.warning("Grade-Tonnage: Failed to get registry instance")
        except Exception as e:
            logger.error(f"Grade-Tonnage: Failed to initialize registry: {e}", exc_info=True)
            self.registry = None
    
    def _check_renderer_layers_for_block_models(self):
        """Check renderer layers for block models that might be displayed."""
        try:
            # Try to get renderer from parent/main window
            parent = self.parent()
            while parent:
                if hasattr(parent, 'viewer_widget') and hasattr(parent.viewer_widget, 'renderer'):
                    renderer = parent.viewer_widget.renderer
                    if hasattr(renderer, 'active_layers') and renderer.active_layers:
                        # Look for block model layers (SGSIM, kriging, etc.)
                        for layer_name, layer_info in renderer.active_layers.items():
                            layer_type = layer_info.get('type', '')
                            layer_data = layer_info.get('data', None)
                            
                            # Check if it's a block model layer
                            if layer_type in ('blocks', 'volume') and layer_data is not None:
                                # If we don't have block model data yet, try to use this one
                                if self._block_model_data is None:
                                    if hasattr(parent, '_extract_block_model_from_grid'):
                                        block_model = parent._extract_block_model_from_grid(layer_data, layer_name)
                                        if block_model is not None:
                                            self._on_bm_loaded(block_model)
                                            logger.info(f"Grade-Tonnage: Found block model in renderer layer: {layer_name}")
                                            return  # Found one, stop searching
                    break
                parent = parent.parent() if parent else None
        except Exception as e:
            logger.debug(f"Grade-Tonnage: Could not check renderer layers: {e}")

    def setup_ui(self):
        """Set up the comprehensive geostatistical grade-tonnage analysis UI."""
        try:
            # Use BasePanel's main_layout if available, otherwise create new layout
            if hasattr(self, 'main_layout') and self.main_layout is not None:
                layout = self.main_layout
                # Clear any existing widgets from base layout
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                layout = QVBoxLayout(self)
                if hasattr(self, 'main_layout'):
                    self.main_layout = layout
            layout.setContentsMargins(5, 5, 5, 5)

            # Main splitter
            main_splitter = QSplitter(Qt.Orientation.Horizontal)

            # LEFT PANEL - Configuration
            left_panel = self._create_configuration_panel()
            main_splitter.addWidget(left_panel)

            # RIGHT PANEL - Results and Visualization
            right_panel = self._create_results_panel()
            main_splitter.addWidget(right_panel)

            # Set splitter proportions (config panel : results panel)
            # Equal 50/50 split for balanced layout
            main_splitter.setStretchFactor(0, 1)
            main_splitter.setStretchFactor(1, 1)
            main_splitter.setSizes([600, 600])  # Equal initial sizes

            layout.addWidget(main_splitter)

        except Exception as e:
            logger.error(f"Grade-Tonnage: Failed to setup UI: {e}", exc_info=True)
            # Create a minimal error message widget as fallback
            error_label = QLabel(f"Error setting up panel: {str(e)}")
            error_label.setStyleSheet("color: red; padding: 20px;")
            if hasattr(self, 'main_layout') and self.main_layout is not None:
                self.main_layout.addWidget(error_label)
            else:
                layout = QVBoxLayout(self)
                layout.addWidget(error_label)

    def _create_configuration_panel(self) -> QWidget:
        """Create the configuration panel with all analysis parameters."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # Scroll area for configuration
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)

        # Data Configuration Group
        self._create_data_config_group(config_layout)

        # Geostatistics Configuration Group
        self._create_geostats_config_group(config_layout)

        # Economic Parameters Group
        self._create_economic_config_group(config_layout)

        # Advanced Economics Group (Multi-Period DCF)
        self._create_advanced_economics_group(config_layout)

        # Analysis Options Group
        self._create_analysis_options_group(config_layout)

        # Advanced Analysis Group (JORC/SAMREC Compliance)
        self._create_advanced_analysis_group(config_layout)

        config_layout.addStretch()
        scroll.setWidget(config_widget)
        layout.addWidget(scroll)

        return panel

    def _create_advanced_analysis_group(self, layout):
        """Create advanced analysis options for JORC/SAMREC compliance."""
        group = QGroupBox("5. Advanced Analysis (JORC/SAMREC)")
        group.setCheckable(True)
        group.setChecked(False)  # Collapsed by default to reduce clutter
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #c62828;
                border: 2px solid #c62828;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #c62828;
                font-weight: bold;
            }
            QGroupBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # Domain Analysis
        domain_layout = QHBoxLayout()
        self.enable_domain_analysis = QCheckBox("Enable Domain Analysis")
        self.enable_domain_analysis.setToolTip(
            "Generate separate GT curves for each geological domain.\n"
            "Required for JORC/SAMREC compliance when domains are present."
        )
        domain_layout.addWidget(self.enable_domain_analysis)
        
        self.domain_column = QComboBox()
        self.domain_column.setToolTip("Column containing domain/lithology codes")
        self.domain_column.setEnabled(False)
        domain_layout.addWidget(QLabel("Domain Col:"))
        domain_layout.addWidget(self.domain_column)
        
        self.enable_domain_analysis.toggled.connect(self.domain_column.setEnabled)
        self.enable_domain_analysis.toggled.connect(self._toggle_domain_tab)
        form_layout.addRow("Domains:", domain_layout)

        # Classification Analysis
        class_layout = QHBoxLayout()
        self.enable_classification_analysis = QCheckBox("Enable Classification Analysis")
        self.enable_classification_analysis.setToolTip(
            "Generate separate GT curves for Measured/Indicated/Inferred.\n"
            "Required for JORC/SAMREC resource reporting."
        )
        class_layout.addWidget(self.enable_classification_analysis)

        self.classification_column = QComboBox()
        self.classification_column.setToolTip("Column containing resource classification")
        self.classification_column.setEnabled(False)
        class_layout.addWidget(QLabel("Class Col:"))
        class_layout.addWidget(self.classification_column)

        self.enable_classification_analysis.toggled.connect(self.classification_column.setEnabled)
        self.enable_classification_analysis.toggled.connect(self._toggle_classification_tab)
        form_layout.addRow("Classification:", class_layout)

        # SGS Uncertainty (future integration)
        sgs_layout = QHBoxLayout()
        self.enable_sgs_uncertainty = QCheckBox("SGS Uncertainty Analysis")
        self.enable_sgs_uncertainty.setToolTip(
            "Compute empirical confidence intervals from SGS realisations.\n"
            "Requires pre-computed SGS results in physical grade space."
        )
        self.enable_sgs_uncertainty.setEnabled(False)  # Requires SGS data
        sgs_layout.addWidget(self.enable_sgs_uncertainty)
        
        sgs_info = QLabel("(requires SGS realisations)")
        sgs_info.setStyleSheet("color: #888; font-style: italic;")
        sgs_layout.addWidget(sgs_info)
        sgs_layout.addStretch()
        
        form_layout.addRow("Uncertainty:", sgs_layout)

        # Info label
        info_label = QLabel(
            "ℹ️ Advanced analysis produces JORC/SAMREC compliant reports\n"
            "with domain-wise and classification stratified GT curves."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setWordWrap(True)
        form_layout.addRow("", info_label)

        layout.addWidget(group)

    def _create_data_config_group(self, layout):
        """Create data configuration group."""
        group = QGroupBox("1. Data Configuration")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #1976d2;
                border: 2px solid #1976d2;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #1976d2;
                font-weight: bold;
            }
        """)

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # Data source selector (for choosing between regular and classified block models)
        self.data_source_box = QComboBox()
        self.data_source_box.setToolTip("Select block model source (regular or classified)")
        self.data_source_box.currentIndexChanged.connect(self._on_data_source_changed)
        form_layout.addRow("Data Source:", self.data_source_box)

        # Multi-source comparison widget
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        form_layout.addRow("", self._source_selection_widget)

        # Grade column
        self.grade_col = QComboBox()
        self.grade_col.setToolTip("Select the grade column for analysis")
        self.grade_col.currentTextChanged.connect(self._auto_detect_cutoff_range)
        form_layout.addRow("Grade Column:", self.grade_col)

        # Tonnage column
        self.tonnage_col = QComboBox()
        self.tonnage_col.setToolTip("Select the tonnage column (or calculate from dimensions)")
        form_layout.addRow("Tonnage Column:", self.tonnage_col)

        # Coordinate columns
        self.x_col = QComboBox()
        self.x_col.setToolTip("X coordinate column for geostatistics")
        form_layout.addRow("X Coordinate:", self.x_col)

        self.y_col = QComboBox()
        self.y_col.setToolTip("Y coordinate column for geostatistics")
        form_layout.addRow("Y Coordinate:", self.y_col)

        self.z_col = QComboBox()
        self.z_col.setToolTip("Z coordinate column for geostatistics")
        form_layout.addRow("Z Coordinate:", self.z_col)

        layout.addWidget(group)

    def _create_geostats_config_group(self, layout):
        """Create geostatistics configuration group."""
        group = QGroupBox("2. Geostatistics Configuration")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #388e3c;
                border: 2px solid #388e3c;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #388e3c;
                font-weight: bold;
            }
        """)

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # DATA MODE SELECTION (CRITICAL FOR CORRECT ANALYSIS)
        data_mode_layout = QHBoxLayout()
        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems(["Block Model", "Composites"])
        self.data_mode_combo.setCurrentIndex(0)  # Default to Block Model
        self.data_mode_combo.setToolTip(
            "BLOCK MODEL: For kriged/estimated block models (no declustering applied)\n"
            "COMPOSITES: For drillhole composites (cell declustering applied)"
        )
        self.data_mode_combo.currentIndexChanged.connect(self._on_data_mode_changed)
        data_mode_layout.addWidget(self.data_mode_combo)
        
        # Add info icon/label
        mode_info_label = QLabel("ⓘ")
        mode_info_label.setToolTip(
            "IMPORTANT: Select the correct data type!\n\n"
            "• BLOCK MODEL: Input is an estimated/kriged block model on a regular grid.\n"
            "  Tonnage = sum of block tonnages above cutoff.\n"
            "  No declustering is applied (blocks are spatially regular).\n\n"
            "• COMPOSITES: Input is drillhole composite samples.\n"
            "  Cell-based declustering corrects for spatial clustering.\n"
            "  Tonnage is derived from declustered proportions × total tonnage."
        )
        mode_info_label.setStyleSheet("color: #1976d2; font-weight: bold;")
        data_mode_layout.addWidget(mode_info_label)
        data_mode_layout.addStretch()
        
        form_layout.addRow("Data Type:", data_mode_layout)

        # Grade weighting method (for block model mode)
        grade_weight_layout = QHBoxLayout()
        self.grade_weighting_combo = QComboBox()
        self.grade_weighting_combo.addItems(["Tonnage-Weighted", "Equal-Weighted"])
        self.grade_weighting_combo.setCurrentIndex(0)  # Default to tonnage-weighted (recommended)
        self.grade_weighting_combo.setToolTip(
            "TONNAGE-WEIGHTED (Recommended): Average grade weighted by block tonnage.\n"
            "  g̅ = Σ(grade_i × tonnage_i) / Σ(tonnage_i)\n"
            "  Best for irregular grids or varying density.\n\n"
            "EQUAL-WEIGHTED: Simple arithmetic average of grades.\n"
            "  g̅ = Σ(grade_i) / n\n"
            "  Appropriate for perfectly regular uniform grids."
        )
        grade_weight_layout.addWidget(self.grade_weighting_combo)
        
        grade_weight_info = QLabel("ⓘ")
        grade_weight_info.setToolTip(
            "Grade weighting method affects the average grade calculation.\n\n"
            "For block models with variable tonnage (irregular grid, varying density),\n"
            "TONNAGE-WEIGHTED is geostatistically correct.\n\n"
            "For perfectly uniform regular grids, both methods give identical results."
        )
        grade_weight_info.setStyleSheet("color: #388e3c; font-weight: bold;")
        grade_weight_layout.addWidget(grade_weight_info)
        grade_weight_layout.addStretch()
        
        form_layout.addRow("Grade Weighting:", grade_weight_layout)

        # Cutoff range
        cutoff_layout = QHBoxLayout()
        self.cutoff_min = QDoubleSpinBox()
        self.cutoff_min.setRange(0.0, 100.0)
        self.cutoff_min.setValue(0.0)
        self.cutoff_min.setDecimals(2)
        cutoff_layout.addWidget(QLabel("Min:"))
        cutoff_layout.addWidget(self.cutoff_min)

        self.cutoff_max = QDoubleSpinBox()
        self.cutoff_max.setRange(0.0, 100.0)
        self.cutoff_max.setValue(5.0)
        self.cutoff_max.setDecimals(2)
        cutoff_layout.addWidget(QLabel("Max:"))
        cutoff_layout.addWidget(self.cutoff_max)

        self.cutoff_steps = QSpinBox()
        self.cutoff_steps.setRange(10, 1000)
        self.cutoff_steps.setValue(50)
        cutoff_layout.addWidget(QLabel("Steps:"))
        cutoff_layout.addWidget(self.cutoff_steps)

        # Auto-detect button for cutoff range
        self.auto_cutoff_btn = QPushButton("Auto")
        self.auto_cutoff_btn.setToolTip("Auto-detect cutoff range from grade data (P5-P95)")
        self.auto_cutoff_btn.setMaximumWidth(50)
        self.auto_cutoff_btn.clicked.connect(self._auto_detect_cutoff_range)
        cutoff_layout.addWidget(self.auto_cutoff_btn)

        form_layout.addRow("Cutoff Range:", cutoff_layout)

        # Declustering options (only applicable for COMPOSITES mode)
        decluster_layout = QHBoxLayout()
        self.use_declustering = QCheckBox("Enable Declustering")
        self.use_declustering.setChecked(False)  # Default off for block model mode
        self.use_declustering.setEnabled(False)  # Disabled initially for block model mode
        self.use_declustering.setToolTip(
            "Cell-based declustering for composite samples.\n\n"
            "⚠️ DECLUSTERING IS ONLY AVAILABLE IN 'COMPOSITES' MODE.\n\n"
            "To enable declustering:\n"
            "1. Change 'Data Type' dropdown to 'Composites'\n"
            "2. The declustering checkbox will automatically enable\n\n"
            "Why? Declustering corrects for spatial clustering in drillhole samples.\n"
            "Block models are already on a regular grid and don't need declustering."
        )
        decluster_layout.addWidget(self.use_declustering)
        
        # Add a status label to show why it's disabled
        self.declustering_status_label = QLabel("(Switch to 'Composites' mode to enable)")
        self.declustering_status_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        decluster_layout.addWidget(self.declustering_status_label)

        self.cell_size = QDoubleSpinBox()
        self.cell_size.setRange(5.0, 1000.0)
        self.cell_size.setValue(25.0)
        self.cell_size.setSuffix(" m")
        self.cell_size.setEnabled(False)  # Disabled initially for block model mode
        self.cell_size.setToolTip("Cell size for declustering grid")
        decluster_layout.addWidget(QLabel("Cell Size:"))
        decluster_layout.addWidget(self.cell_size)

        form_layout.addRow("Declustering:", decluster_layout)

        # Confidence intervals (heuristic CV-based)
        self.show_confidence = QCheckBox("Show Uncertainty Bands (±CV)")
        self.show_confidence.setChecked(True)
        self.show_confidence.setToolTip(
            "Display heuristic ±CV uncertainty bands on plots.\n"
            "NOTE: These are NOT formal statistical confidence intervals.\n"
            "For proper geostatistical uncertainty, use SGS realisations."
        )
        form_layout.addRow("Uncertainty:", self.show_confidence)

        layout.addWidget(group)
    
    def _on_data_mode_changed(self, index: int):
        """Handle data mode selection change."""
        is_composites = (index == 1)  # Composites mode
        
        # Enable/disable declustering controls based on mode
        self.use_declustering.setEnabled(is_composites)
        self.cell_size.setEnabled(is_composites)
        
        # Update status label
        if hasattr(self, 'declustering_status_label'):
            if is_composites:
                self.declustering_status_label.setText("(Available in Composites mode)")
                self.declustering_status_label.setStyleSheet("color: #388e3c; font-size: 9px; font-style: italic;")
            else:
                self.declustering_status_label.setText("(Switch to 'Composites' mode to enable)")
                self.declustering_status_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        
        if is_composites:
            self.use_declustering.setChecked(True)
            self.status_text.setPlainText(
                "📊 COMPOSITES mode selected.\n"
                "Cell-based declustering will be applied to correct for spatial clustering.\n"
                "Tonnage will be derived from declustered proportions × total deposit tonnage.\n\n"
                "✅ Declustering is now ENABLED and ready to use."
            )
        else:
            self.use_declustering.setChecked(False)
            self.status_text.setPlainText(
                "🧊 BLOCK MODEL mode selected.\n"
                "No declustering applied (block models are spatially regular).\n"
                "Tonnage = sum of block tonnages above each cutoff.\n\n"
                "ℹ️ To use declustering, switch 'Data Type' to 'Composites' mode."
            )

    def _create_economic_config_group(self, layout):
        """Create economic parameters configuration group."""
        group = QGroupBox("3. Economic Parameters (click to expand)")
        group.setCheckable(True)
        group.setChecked(True)  # Expanded by default for basic analysis
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #f57c00;
                border: 2px solid #f57c00;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #f57c00;
                font-weight: bold;
            }
            QGroupBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # Metal price
        self.metal_price = QDoubleSpinBox()
        self.metal_price.setRange(0.0, 10000.0)
        self.metal_price.setValue(50.0)
        self.metal_price.setPrefix("$/")
        self.metal_price.setSuffix("unit")
        form_layout.addRow("Metal Price:", self.metal_price)

        # Operating costs
        cost_layout = QHBoxLayout()
        self.mining_cost = QDoubleSpinBox()
        self.mining_cost.setRange(0.0, 1000.0)
        self.mining_cost.setValue(15.0)
        self.mining_cost.setSuffix(" $/t")
        cost_layout.addWidget(QLabel("Mining:"))
        cost_layout.addWidget(self.mining_cost)

        self.processing_cost = QDoubleSpinBox()
        self.processing_cost.setRange(0.0, 1000.0)
        self.processing_cost.setValue(25.0)
        self.processing_cost.setSuffix(" $/t")
        cost_layout.addWidget(QLabel("Processing:"))
        cost_layout.addWidget(self.processing_cost)

        form_layout.addRow("Operating Costs:", cost_layout)

        # Recovery and discount
        param_layout = QHBoxLayout()
        self.recovery = QDoubleSpinBox()
        self.recovery.setRange(0.0, 1.0)
        self.recovery.setValue(0.85)
        self.recovery.setSuffix(" %")
        self.recovery.setSingleStep(0.01)
        param_layout.addWidget(QLabel("Recovery:"))
        param_layout.addWidget(self.recovery)

        self.discount_rate = QDoubleSpinBox()
        self.discount_rate.setRange(0.0, 0.5)
        self.discount_rate.setValue(0.10)
        self.discount_rate.setSuffix(" %")
        self.discount_rate.setSingleStep(0.01)
        param_layout.addWidget(QLabel("Discount Rate:"))
        param_layout.addWidget(self.discount_rate)

        form_layout.addRow("Process Parameters:", param_layout)

        layout.addWidget(group)

    def _create_advanced_economics_group(self, layout):
        """Create advanced multi-period economics configuration group (collapsible)."""
        group = QGroupBox("3b. Advanced Economics (Multi-Period DCF)")
        group.setCheckable(True)
        group.setChecked(False)  # Collapsed by default
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #c62828;
                border: 2px solid #c62828;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #c62828;
                font-weight: bold;
            }
            QGroupBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        group.setToolTip(
            "Enable multi-period DCF analysis for audit-grade NPV optimization.\n"
            "Includes mine capacity constraints, capital expenditure, and taxes.\n"
            "Required for proper 'NPV-optimised cutoff' claims to auditors."
        )

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # Mine capacity
        capacity_layout = QHBoxLayout()
        self.annual_capacity = QDoubleSpinBox()
        self.annual_capacity.setRange(100_000, 500_000_000)
        self.annual_capacity.setValue(10_000_000)
        self.annual_capacity.setDecimals(0)
        self.annual_capacity.setSuffix(" t/year")
        self.annual_capacity.setToolTip("Maximum annual ore processing capacity")
        capacity_layout.addWidget(self.annual_capacity)
        
        self.ramp_up_years = QSpinBox()
        self.ramp_up_years.setRange(1, 10)
        self.ramp_up_years.setValue(2)
        self.ramp_up_years.setToolTip("Years to reach full capacity")
        capacity_layout.addWidget(QLabel("Ramp-up:"))
        capacity_layout.addWidget(self.ramp_up_years)
        capacity_layout.addWidget(QLabel("years"))
        
        form_layout.addRow("Annual Capacity:", capacity_layout)

        # Capital expenditure
        capex_layout = QHBoxLayout()
        self.initial_capex = QDoubleSpinBox()
        self.initial_capex.setRange(0, 50_000_000_000)
        self.initial_capex.setValue(500_000_000)
        self.initial_capex.setDecimals(0)
        self.initial_capex.setPrefix("$")
        self.initial_capex.setToolTip("Initial capital expenditure before production")
        capex_layout.addWidget(self.initial_capex)
        
        self.sustaining_capex = QDoubleSpinBox()
        self.sustaining_capex.setRange(0.0, 100.0)
        self.sustaining_capex.setValue(2.0)
        self.sustaining_capex.setSuffix(" $/t")
        self.sustaining_capex.setToolTip("Annual sustaining capital per tonne ore")
        capex_layout.addWidget(QLabel("Sustaining:"))
        capex_layout.addWidget(self.sustaining_capex)
        
        form_layout.addRow("Capital Expenditure:", capex_layout)

        # Closure cost
        self.closure_cost = QDoubleSpinBox()
        self.closure_cost.setRange(0, 5_000_000_000)
        self.closure_cost.setValue(50_000_000)
        self.closure_cost.setDecimals(0)
        self.closure_cost.setPrefix("$")
        self.closure_cost.setToolTip("Mine closure and rehabilitation cost")
        form_layout.addRow("Closure Cost:", self.closure_cost)

        # Tax parameters
        tax_layout = QHBoxLayout()
        self.tax_rate = QDoubleSpinBox()
        self.tax_rate.setRange(0.0, 0.5)
        self.tax_rate.setValue(0.30)
        self.tax_rate.setSingleStep(0.01)
        self.tax_rate.setToolTip("Corporate income tax rate")
        tax_layout.addWidget(QLabel("Tax Rate:"))
        tax_layout.addWidget(self.tax_rate)
        
        self.royalty_rate = QDoubleSpinBox()
        self.royalty_rate.setRange(0.0, 0.20)
        self.royalty_rate.setValue(0.05)
        self.royalty_rate.setSingleStep(0.01)
        self.royalty_rate.setToolTip("Royalty rate on revenue")
        tax_layout.addWidget(QLabel("Royalty:"))
        tax_layout.addWidget(self.royalty_rate)
        
        form_layout.addRow("Tax/Royalty:", tax_layout)

        # Mining modifying factors
        mmf_layout = QHBoxLayout()
        self.dilution = QDoubleSpinBox()
        self.dilution.setRange(0.0, 0.5)
        self.dilution.setValue(0.05)
        self.dilution.setSingleStep(0.01)
        self.dilution.setToolTip("Mining dilution factor")
        mmf_layout.addWidget(QLabel("Dilution:"))
        mmf_layout.addWidget(self.dilution)
        
        self.mining_loss = QDoubleSpinBox()
        self.mining_loss.setRange(0.0, 0.3)
        self.mining_loss.setValue(0.02)
        self.mining_loss.setSingleStep(0.01)
        self.mining_loss.setToolTip("Mining loss factor")
        mmf_layout.addWidget(QLabel("Loss:"))
        mmf_layout.addWidget(self.mining_loss)
        
        form_layout.addRow("Mining Factors:", mmf_layout)

        # Info label about multi-period NPV
        info_label = QLabel(
            "ℹ️ Multi-period DCF generates annual cash flows with capacity constraints,\n"
            "   proper discounting, tax, and depreciation for audit-ready NPV analysis."
        )
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        form_layout.addRow(info_label)

        layout.addWidget(group)
        
        # Store reference for toggling
        self.advanced_economics_group = group

    def _create_analysis_options_group(self, layout):
        """Create analysis options group."""
        group = QGroupBox("4. Analysis Options")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #7b1fa2;
                border: 2px solid #7b1fa2;
                border-radius: 5px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #7b1fa2;
                font-weight: bold;
            }
        """)

        form_layout = QFormLayout(group)
        form_layout.setSpacing(8)

        # Optimization method
        self.optimization_method = QComboBox()
        self.optimization_method.addItems([
            "NPV Maximization",
            "IRR Maximization",
            "Payback Minimization",
            "Risk-Adjusted NPV",
            "Multi-Criteria"
        ])
        self.optimization_method.setCurrentText("NPV Maximization")
        form_layout.addRow("Optimization Method:", self.optimization_method)

        # Plot options
        plot_layout = QHBoxLayout()
        self.interactive_plots = QCheckBox("Interactive Plots")
        self.interactive_plots.setChecked(False)
        self.interactive_plots.setToolTip("Use interactive Plotly plots (requires plotly installation)")
        plot_layout.addWidget(self.interactive_plots)

        self.color_scheme = QComboBox()
        self.color_scheme.addItems(["Geostats", "Default", "High Contrast"])
        self.color_scheme.setCurrentText("Geostats")
        plot_layout.addWidget(QLabel("Color Scheme:"))
        plot_layout.addWidget(self.color_scheme)

        form_layout.addRow("Visualization:", plot_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        self.run_analysis_btn = QPushButton("🚀 Run Geostatistical Analysis")
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.run_analysis_btn.clicked.connect(self._run_geostats_analysis)
        self.run_analysis_btn.setEnabled(False)  # Disabled until data is loaded
        button_layout.addWidget(self.run_analysis_btn)

        self.clear_btn = QPushButton("🗑️ Clear Results")
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #757575;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            QPushButton:pressed {
                background-color: #424242;
            }
            QPushButton:disabled {
                background-color: #bdbdbd;
                color: #757575;
            }
        """)
        self.clear_btn.clicked.connect(self._clear_all_results)
        self.clear_btn.setEnabled(False)  # Disabled until results exist
        self.clear_btn.setToolTip("Clear all analysis results, tables, and visualizations")
        button_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("📊 Export Results")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976d2;
            }
        """)
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)

        form_layout.addRow("", button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        form_layout.addRow("Progress:", self.progress_bar)

        layout.addWidget(group)

    def _create_results_panel(self) -> QWidget:
        """Create the results and visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # Results tabs
        self.results_tabs = QTabWidget()

        # Core tabs (always visible)
        # Grade-Tonnage Curve Tab
        curve_tab = self._create_curve_tab()
        self.results_tabs.addTab(curve_tab, "📈 Grade-Tonnage Curve")

        # Sensitivity Analysis Tab
        sensitivity_tab = self._create_sensitivity_tab()
        self.results_tabs.addTab(sensitivity_tab, "🎯 Cut-off Sensitivity")

        # Statistics Tab
        stats_tab = self._create_statistics_tab()
        self.results_tabs.addTab(stats_tab, "📋 Statistics")

        # Advanced tabs (hidden until enabled via checkboxes)
        # These are created but stored for later addition
        self._domain_tab = self._create_domain_analysis_tab()
        self._classification_tab = self._create_classification_tab()
        self._dashboard_tab = self._create_dashboard_tab()

        # Track which advanced tabs are currently shown
        self._domain_tab_shown = False
        self._classification_tab_shown = False
        self._dashboard_tab_shown = False

        layout.addWidget(self.results_tabs)

        # Popup buttons row
        popup_layout = QHBoxLayout()
        popup_layout.addWidget(QLabel("Open plots in separate windows:"))
        
        self.popup_curve_btn = QPushButton("📈 Curve")
        self.popup_curve_btn.clicked.connect(lambda: self._open_plot_window("curve"))
        self.popup_curve_btn.setEnabled(False)
        popup_layout.addWidget(self.popup_curve_btn)
        
        self.popup_sensitivity_btn = QPushButton("🎯 Sensitivity")
        self.popup_sensitivity_btn.clicked.connect(lambda: self._open_plot_window("sensitivity"))
        self.popup_sensitivity_btn.setEnabled(False)
        popup_layout.addWidget(self.popup_sensitivity_btn)
        
        self.popup_dashboard_btn = QPushButton("📊 Dashboard")
        self.popup_dashboard_btn.clicked.connect(lambda: self._open_plot_window("dashboard"))
        self.popup_dashboard_btn.setEnabled(False)
        popup_layout.addWidget(self.popup_dashboard_btn)
        
        self.popup_all_btn = QPushButton("🖼️ All Plots")
        self.popup_all_btn.clicked.connect(lambda: self._open_plot_window("all"))
        self.popup_all_btn.setEnabled(False)
        self.popup_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196f3;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover { background-color: #1976d2; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        popup_layout.addWidget(self.popup_all_btn)
        
        popup_layout.addStretch()
        layout.addLayout(popup_layout)

        # Status bar
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("Ready to analyze geostatistical grade-tonnage curves...")
        layout.addWidget(self.status_text)

        return panel

    def _create_curve_tab(self) -> QWidget:
        """Create the grade-tonnage curve visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Toolbar container (for NavigationToolbar + export button)
        self.curve_toolbar_container = QWidget()
        self.curve_toolbar_layout = QHBoxLayout(self.curve_toolbar_container)
        self.curve_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_toolbar_layout.addStretch()  # Will be replaced when canvas is created

        # Export button (always visible)
        self.curve_export_btn = QPushButton("Export Chart")
        self.curve_export_btn.setToolTip("Export chart as PNG, SVG, or PDF")
        self.curve_export_btn.clicked.connect(lambda: self._export_chart("curve"))
        self.curve_export_btn.setEnabled(False)
        self.curve_toolbar_layout.addWidget(self.curve_export_btn)

        layout.addWidget(self.curve_toolbar_container)

        # Plot area
        self.curve_canvas = None
        self.curve_toolbar = None  # NavigationToolbar
        self.curve_placeholder = QLabel("Run analysis to display grade-tonnage curve")
        self.curve_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.curve_placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.curve_placeholder)

        # Curve data table
        self.curve_table = QTableWidget(0, 9)
        self.curve_table.setHorizontalHeaderLabels([
            "Cutoff", "Tonnage", "Avg Grade", "Metal Qty", "Net Value",
            "CV Lower", "CV Upper", "CV Factor", "Decluster Wt"
        ])
        self.curve_table.horizontalHeader().setStretchLastSection(True)
        self.curve_table.setVisible(False)
        layout.addWidget(self.curve_table)

        return tab

    def _create_sensitivity_tab(self) -> QWidget:
        """Create the sensitivity analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Toolbar container (for NavigationToolbar + export button)
        self.sensitivity_toolbar_container = QWidget()
        self.sensitivity_toolbar_layout = QHBoxLayout(self.sensitivity_toolbar_container)
        self.sensitivity_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.sensitivity_toolbar_layout.addStretch()

        # Export button
        self.sensitivity_export_btn = QPushButton("Export Chart")
        self.sensitivity_export_btn.setToolTip("Export chart as PNG, SVG, or PDF")
        self.sensitivity_export_btn.clicked.connect(lambda: self._export_chart("sensitivity"))
        self.sensitivity_export_btn.setEnabled(False)
        self.sensitivity_toolbar_layout.addWidget(self.sensitivity_export_btn)

        layout.addWidget(self.sensitivity_toolbar_container)

        # Plot area
        self.sensitivity_canvas = None
        self.sensitivity_toolbar = None  # NavigationToolbar
        self.sensitivity_placeholder = QLabel("Run analysis to display sensitivity analysis")
        self.sensitivity_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensitivity_placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.sensitivity_placeholder)

        # Optimal cutoff display
        self.optimal_display = QLabel()
        self.optimal_display.setVisible(False)
        self.optimal_display.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                border: 2px solid #4caf50;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
                color: #2e7d32;
            }
        """)
        layout.addWidget(self.optimal_display)

        return tab

    def _create_dashboard_tab(self) -> QWidget:
        """Create the multi-criteria dashboard tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Dashboard plot area
        self.dashboard_canvas = None
        self.dashboard_placeholder = QLabel("Run analysis to display comprehensive dashboard")
        self.dashboard_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.dashboard_placeholder)

        return tab

    def _create_statistics_tab(self) -> QWidget:
        """Create the statistics and summary tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Statistics text area
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("Run analysis to display statistical summary...")
        layout.addWidget(self.stats_text)

        # Data validation results
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        self.validation_text.setMaximumHeight(150)
        self.validation_text.setPlainText("Data validation results will appear here...")
        layout.addWidget(self.validation_text)

        return tab

    def _create_domain_analysis_tab(self) -> QWidget:
        """Create the domain analysis tab for JORC/SAMREC compliance."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("🏔️ Domain-wise Grade-Tonnage Analysis")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #c62828;")
        layout.addWidget(header)

        info = QLabel(
            "Separate GT curves for each geological domain (lithology, ore type, etc.).\n"
            "Required for JORC/SAMREC compliance when domains are present."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Domain results placeholder
        self.domain_canvas = None
        self.domain_placeholder = QLabel("Enable Domain Analysis and run to display results")
        self.domain_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.domain_placeholder.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 12px;
                padding: 40px;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.domain_placeholder)

        # Domain summary table
        self.domain_summary_table = QTableWidget(0, 4)
        self.domain_summary_table.setHorizontalHeaderLabels([
            "Domain", "Tonnage (t)", "Mean Grade", "Metal Content"
        ])
        self.domain_summary_table.horizontalHeader().setStretchLastSection(True)
        self.domain_summary_table.setVisible(False)
        layout.addWidget(self.domain_summary_table)

        # Domain results text
        self.domain_results_text = QTextEdit()
        self.domain_results_text.setReadOnly(True)
        self.domain_results_text.setMaximumHeight(150)
        self.domain_results_text.setPlainText("Domain analysis results will appear here...")
        layout.addWidget(self.domain_results_text)

        return tab

    def _create_classification_tab(self) -> QWidget:
        """Create the classification analysis tab for JORC/SAMREC compliance."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Header
        header = QLabel("📊 Resource Classification Analysis (M/I/I)")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2e7d32;")
        layout.addWidget(header)

        info = QLabel(
            "Separate GT curves for Measured, Indicated, and Inferred resources.\n"
            "Required for JORC 2012 / SAMREC compliant resource reporting."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        # Classification results placeholder
        self.classification_canvas = None
        self.classification_placeholder = QLabel("Enable Classification Analysis and run to display results")
        self.classification_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.classification_placeholder.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 12px;
                padding: 40px;
                border: 2px dashed #ccc;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.classification_placeholder)

        # Classification summary
        summary_group = QGroupBox("Resource Summary by Classification")
        summary_layout = QFormLayout(summary_group)
        
        self.measured_summary = QLabel("--")
        self.measured_summary.setStyleSheet("color: #2ca02c; font-weight: bold;")  # Green
        summary_layout.addRow("🟢 Measured:", self.measured_summary)
        
        self.indicated_summary = QLabel("--")
        self.indicated_summary.setStyleSheet("color: #ffbf00; font-weight: bold;")  # Amber
        summary_layout.addRow("🟡 Indicated:", self.indicated_summary)
        
        self.inferred_summary = QLabel("--")
        self.inferred_summary.setStyleSheet("color: #d62728; font-weight: bold;")  # Red
        summary_layout.addRow("🔴 Inferred:", self.inferred_summary)
        
        self.total_resource_summary = QLabel("--")
        self.total_resource_summary.setStyleSheet("font-weight: bold;")
        summary_layout.addRow("📊 Total:", self.total_resource_summary)
        
        layout.addWidget(summary_group)

        # Classification results text
        self.classification_results_text = QTextEdit()
        self.classification_results_text.setReadOnly(True)
        self.classification_results_text.setMaximumHeight(150)
        self.classification_results_text.setPlainText("Classification analysis results will appear here...")
        layout.addWidget(self.classification_results_text)

        return tab

    def _update_column_selectors(self):
        """Update all column selector combo boxes with available columns from block model."""
        if self._block_model_data is None or len(self._block_model_data) == 0:
            return

        try:
            # Get all numeric columns for grade/tonnage
            numeric_cols = self._block_model_data.select_dtypes(
                include=[float, 'int64', 'int32', 'int']
            ).columns.tolist()

            # Get all columns
            all_cols = self._block_model_data.columns.tolist()

            # Update grade column selector
            if hasattr(self, 'grade_col'):
                current_grade = self.grade_col.currentText()
                self.grade_col.clear()
                self.grade_col.addItems(numeric_cols)
                # Try to restore previous selection or auto-select likely grade column
                if current_grade and current_grade in numeric_cols:
                    self.grade_col.setCurrentText(current_grade)
                else:
                    # Auto-select likely grade columns
                    for col in numeric_cols:
                        col_upper = col.upper()
                        if any(term in col_upper for term in ['GRADE', 'AU', 'AG', 'CU', 'FE', 'ZN', 'PB']):
                            self.grade_col.setCurrentText(col)
                            break

            # Update tonnage column selector
            if hasattr(self, 'tonnage_col'):
                current_tonnage = self.tonnage_col.currentText()
                self.tonnage_col.clear()
                self.tonnage_col.addItems(numeric_cols)
                # Try to restore or auto-select
                if current_tonnage and current_tonnage in numeric_cols:
                    self.tonnage_col.setCurrentText(current_tonnage)
                else:
                    for col in numeric_cols:
                        col_upper = col.upper()
                        if any(term in col_upper for term in ['TONNAGE', 'TONNES', 'TONS', 'WEIGHT']):
                            self.tonnage_col.setCurrentText(col)
                            break

            # Update coordinate column selectors
            coord_candidates = {
                'x': ['X', 'XCENTRE', 'XCENTER', 'EAST', 'EASTING'],
                'y': ['Y', 'YCENTRE', 'YCENTER', 'NORTH', 'NORTHING'],
                'z': ['Z', 'ZCENTRE', 'ZCENTER', 'RL', 'ELEVATION']
            }

            for coord, selector_name in [('x', 'x_col'), ('y', 'y_col'), ('z', 'z_col')]:
                if hasattr(self, selector_name):
                    selector = getattr(self, selector_name)
                    current = selector.currentText()
                    selector.clear()
                    selector.addItems(numeric_cols)
                    if current and current in numeric_cols:
                        selector.setCurrentText(current)
                    else:
                        for col in numeric_cols:
                            if col.upper() in coord_candidates[coord]:
                                selector.setCurrentText(col)
                                break

            # Update domain and classification column selectors
            if hasattr(self, 'domain_column'):
                self.domain_column.clear()
                self.domain_column.addItems([''] + all_cols)
                for col in all_cols:
                    col_upper = col.upper()
                    if any(term in col_upper for term in ['DOMAIN', 'LITH', 'ROCK', 'ZONE']):
                        self.domain_column.setCurrentText(col)
                        break

            if hasattr(self, 'classification_column'):
                self.classification_column.clear()
                self.classification_column.addItems([''] + all_cols)
                for col in all_cols:
                    col_upper = col.upper()
                    if any(term in col_upper for term in ['CLASS', 'CATEGORY', 'RESOURCE_CLASS']):
                        self.classification_column.setCurrentText(col)
                        break

            # Enable the run button
            if hasattr(self, 'run_analysis_btn'):
                self.run_analysis_btn.setEnabled(True)

            logger.info(f"Grade-Tonnage: Updated column selectors with {len(numeric_cols)} numeric columns")

        except Exception as e:
            logger.error(f"Grade-Tonnage: Failed to update column selectors: {e}", exc_info=True)

    def _toggle_domain_tab(self, enabled: bool):
        """Show or hide the Domain Analysis tab based on checkbox state."""
        if not hasattr(self, 'results_tabs') or not hasattr(self, '_domain_tab'):
            return

        if enabled and not self._domain_tab_shown:
            # Add the domain tab
            self.results_tabs.addTab(self._domain_tab, "🏔️ Domain Analysis")
            self._domain_tab_shown = True
            logger.debug("Grade-Tonnage: Domain Analysis tab shown")
        elif not enabled and self._domain_tab_shown:
            # Find and remove the domain tab
            for i in range(self.results_tabs.count()):
                if self.results_tabs.widget(i) == self._domain_tab:
                    self.results_tabs.removeTab(i)
                    break
            self._domain_tab_shown = False
            logger.debug("Grade-Tonnage: Domain Analysis tab hidden")

    def _toggle_classification_tab(self, enabled: bool):
        """Show or hide the Classification Analysis tab based on checkbox state."""
        if not hasattr(self, 'results_tabs') or not hasattr(self, '_classification_tab'):
            return

        if enabled and not self._classification_tab_shown:
            # Add the classification tab
            self.results_tabs.addTab(self._classification_tab, "📊 M/I/I Classification")
            self._classification_tab_shown = True
            logger.debug("Grade-Tonnage: Classification tab shown")
        elif not enabled and self._classification_tab_shown:
            # Find and remove the classification tab
            for i in range(self.results_tabs.count()):
                if self.results_tabs.widget(i) == self._classification_tab:
                    self.results_tabs.removeTab(i)
                    break
            self._classification_tab_shown = False
            logger.debug("Grade-Tonnage: Classification tab hidden")

    def bind_controller(self, controller):
        """Override to re-initialize registry when controller is bound."""
        super().bind_controller(controller)
        # If registry wasn't initialized before, try again now that we have a controller
        if not self._registry_initialized and controller:
            logger.info("Grade-Tonnage: Controller bound, re-initializing registry connection")
            self._init_registry()
    
    def set_block_model(self, bm) -> None:
        """
        Compatibility helper so external callers (e.g. MainWindow) can inject
        the current block model.

        Accepts either a `BlockModel` instance (with `to_dataframe`) or a
        `pandas.DataFrame` directly and forwards to the internal loader.
        """
        try:
            logger.info(f"Grade-Tonnage: set_block_model called with {type(bm).__name__}")
            self._on_bm_loaded(bm)
        except Exception as e:
            logger.error(f"GradeTonnagePanel.set_block_model failed: {e}", exc_info=True)

    def _on_bm_loaded(self, bm):
        if bm is None:
            logger.warning("Grade-Tonnage: _on_bm_loaded called with None")
            return
            
        try:
            if hasattr(bm, 'to_dataframe'):
                self._block_model_data = bm.to_dataframe()
                logger.info(f"Grade-Tonnage: Converted BlockModel to DataFrame, shape: {self._block_model_data.shape}")
            elif isinstance(bm, pd.DataFrame):
                self._block_model_data = bm
                logger.info(f"Grade-Tonnage: Received DataFrame directly, shape: {self._block_model_data.shape}")
            else:
                logger.warning(f"Grade-Tonnage: Unknown block model type: {type(bm)}")
                return
            
            if self._block_model_data is None or len(self._block_model_data) == 0:
                logger.warning("Grade-Tonnage: Block model data is empty")
                return
            
            # Update column selectors with available columns
            self._update_column_selectors()
        except Exception as e:
            logger.error(f"Grade-Tonnage: Error in _on_bm_loaded: {e}", exc_info=True)

    def _on_block_model_loaded(self, block_model):
        """Handle block model loaded from DataRegistry."""
        logger.info("Grade-Tonnage: Block model received from registry")
        self._stored_block_model = block_model
        if "block_model" not in self._available_sources:
            self._available_sources.append("block_model")
        self._update_data_source_selector()
        # Only load if this is the current source or no data loaded yet
        if self._current_source == "block_model" or self._block_model_data is None:
            self._on_bm_loaded(block_model)

    def _on_block_model_classified(self, block_model):
        """Handle classified block model from DataRegistry."""
        logger.info("Grade-Tonnage: Classified block model received from registry")
        # Store as SEPARATE classified block model (don't overwrite regular block model)
        self._stored_classified_block_model = block_model
        if "classified_block_model" not in self._available_sources:
            self._available_sources.append("classified_block_model")
        self._update_data_source_selector()
        # Auto-switch to classified model when it becomes available
        self._current_source = "classified_block_model"
        if hasattr(self, 'data_source_box'):
            idx = self.data_source_box.findData("classified_block_model")
            if idx >= 0:
                self.data_source_box.setCurrentIndex(idx)
        self._on_bm_loaded(block_model)

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results from DataRegistry - registers individual statistics as separate sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        try:
            import pandas as pd
            import numpy as np
            import pyvista as pv

            if results is None:
                return

            if not isinstance(results, dict):
                logger.warning(f"Grade-Tonnage: SGSIM results is not a dict, type={type(results)}")
                return

            logger.info("Grade-Tonnage: SGSIM results received from registry")
            self._stored_sgsim_results = results
            variable = results.get('variable', 'Grade')
            summary = results.get('summary', {})
            params = results.get('params')
            grid = results.get('grid') or results.get('pyvista_grid')

            logger.info(f"Grade-Tonnage: SGSIM results keys: {list(results.keys())}")
            logger.info(f"Grade-Tonnage: Summary keys: {list(summary.keys()) if summary else 'None'}")
            logger.info(f"Grade-Tonnage: params = {params is not None}")

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
                        logger.info(f"Grade-Tonnage: Extracted {n_blocks:,} cell centers from grid")

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
                    logger.info(f"Grade-Tonnage: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
                except Exception as e:
                    logger.warning(f"Grade-Tonnage: Failed to generate coords from params: {e}")

            if base_df is None or base_df.empty:
                # Fallback: realizations array with grid coordinates
                reals = results.get('realizations') or results.get('realizations_raw')
                if reals is not None:
                    grid_x, grid_y, grid_z = results.get('grid_x'), results.get('grid_y'), results.get('grid_z')
                    if grid_x is not None and isinstance(reals, np.ndarray):
                        mean_estimate = np.mean(reals, axis=0) if reals.ndim == 2 else reals.ravel()
                        df = pd.DataFrame({
                            'X': np.asarray(grid_x).ravel(), 'Y': np.asarray(grid_y).ravel(),
                            'Z': np.asarray(grid_z).ravel(), variable: mean_estimate
                        })
                        self._stored_sgsim_df = df

                        source_key = f"sgsim_etype_{variable}"
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': f"SGSIM E-type Mean ({variable})",
                            'property': variable
                        }
                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)
                        self._update_data_source_selector()
                        logger.info(f"Grade-Tonnage: Registered SGSIM E-type Mean from fallback")
                else:
                    logger.warning("Grade-Tonnage: No grid, params, or realizations found in SGSIM results")
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
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': display_name,
                            'property': prop_name
                        }

                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)

                        found_stats.append(stat_key)
                        logger.info(f"Grade-Tonnage: Registered {display_prefix} ({variable})")

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
                        source_key = f"sgsim_mean_{variable}"
                        display_name = f"SGSIM Mean ({variable}) - {n_blocks:,} blocks"
                    elif 'PROB' in prop_upper:
                        source_key = f"sgsim_prob_{prop_name}"
                        display_name = f"SGSIM Probability ({prop_name}) - {n_blocks:,} blocks"
                    else:
                        source_key = f"sgsim_{prop_name}"
                        display_name = f"SGSIM {prop_name} - {n_blocks:,} blocks"

                    if source_key not in self._block_model_sources:
                        self._block_model_sources[source_key] = {
                            'df': df,
                            'display_name': display_name,
                            'property': prop_name
                        }
                        if source_key not in self._available_sources:
                            self._available_sources.append(source_key)
                        found_stats.append(prop_name)

            # Store combined DataFrame for backward compatibility
            if found_stats:
                df_all = base_df.copy()
                for stat_key in ['mean', 'std', 'p10', 'p50', 'p90']:
                    stat_data = summary.get(stat_key)
                    if stat_data is not None:
                        stat_values = np.asarray(stat_data).flatten()
                        if len(stat_values) == n_blocks:
                            df_all[f"{variable}_{stat_key.upper()}"] = stat_values
                self._stored_sgsim_df = df_all
                logger.info(f"Grade-Tonnage: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

            self._update_data_source_selector()

            # Auto-load if no data yet (prefer Mean)
            if self._block_model_data is None and self._available_sources:
                # Try to find a SGSIM Mean source
                mean_source = next((s for s in self._available_sources if 'mean' in s.lower()), None)
                if mean_source and mean_source in self._block_model_sources:
                    self._current_source = mean_source
                    self._on_bm_loaded(self._block_model_sources[mean_source]['df'])
                    logger.info(f"Grade-Tonnage: Auto-loaded SGSIM Mean")

        except Exception as e:
            logger.warning(f"Grade-Tonnage: Failed to load SGSIM results: {e}", exc_info=True)

    def _update_data_source_selector(self):
        """Update the data source dropdown with available sources (including individual SGSIM stats)."""
        if not hasattr(self, 'data_source_box'):
            return

        # Remember current selection
        current_data = self.data_source_box.currentData()

        # Block signals while updating
        self.data_source_box.blockSignals(True)
        self.data_source_box.clear()

        # Add available sources - regular block models first
        if "block_model" in self._available_sources:
            bm = self._stored_block_model
            count = len(bm) if bm is not None else 0
            self.data_source_box.addItem(f"Block Model ({count:,} blocks)", "block_model")

        if "classified_block_model" in self._available_sources:
            bm = self._stored_classified_block_model
            count = len(bm) if bm is not None else 0
            self.data_source_box.addItem(f"Classified Block Model ({count:,} blocks)", "classified_block_model")

        # Add individual SGSIM sources
        for source_key in self._available_sources:
            if source_key.startswith('sgsim_') and source_key in self._block_model_sources:
                source_info = self._block_model_sources[source_key]
                df = source_info.get('df')
                count = len(df) if df is not None else 0
                display_name = source_info.get('display_name', source_key)
                self.data_source_box.addItem(f"{display_name} ({count:,} blocks)", source_key)

        # Legacy: single SGSIM source (if not using individual stats)
        if "sgsim" in self._available_sources and not any(s.startswith('sgsim_') for s in self._available_sources):
            df = self._stored_sgsim_df
            count = len(df) if df is not None else 0
            self.data_source_box.addItem(f"SGSIM Results ({count:,} blocks)", "sgsim")

        if self.data_source_box.count() == 0:
            self.data_source_box.addItem("No data loaded", "none")

        # Restore selection if possible
        if current_data:
            idx = self.data_source_box.findData(current_data)
            if idx >= 0:
                self.data_source_box.setCurrentIndex(idx)

        self.data_source_box.blockSignals(False)
        logger.debug(f"Grade-Tonnage: Updated data source selector with {len(self._available_sources)} sources")

        # Also update comparison sources widget
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()

    def _on_data_source_changed(self, index: int):
        """Handle data source selection change."""
        if index < 0 or not hasattr(self, 'data_source_box'):
            return

        source_type = self.data_source_box.currentData()
        if source_type is None or source_type == "none":
            return

        logger.info(f"Grade-Tonnage: Data source changed to {source_type}")
        self._current_source = source_type

        if source_type == "block_model":
            if self._stored_block_model is not None:
                self._on_bm_loaded(self._stored_block_model)
                logger.info("Grade-Tonnage: Switched to regular block model")
        elif source_type == "classified_block_model":
            if self._stored_classified_block_model is not None:
                self._on_bm_loaded(self._stored_classified_block_model)
                logger.info("Grade-Tonnage: Switched to classified block model")
        elif source_type == "sgsim":
            if hasattr(self, '_stored_sgsim_df') and self._stored_sgsim_df is not None:
                self._on_bm_loaded(self._stored_sgsim_df)
                logger.info("Grade-Tonnage: Switched to SGSIM results")
        elif source_type.startswith('sgsim_') and source_type in self._block_model_sources:
            # Handle individual SGSIM statistics
            source_info = self._block_model_sources[source_type]
            df = source_info.get('df')
            if df is not None:
                self._on_bm_loaded(df)
                logger.info(f"Grade-Tonnage: Switched to {source_info.get('display_name', source_type)}")

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        if enabled:
            self._update_comparison_sources()
            # Disable the single data source selector in comparison mode
            self.data_source_box.setEnabled(False)
        else:
            self.data_source_box.setEnabled(True)
            self._comparison_results.clear()
        logger.info(f"Grade-Tonnage: Comparison mode {'enabled' if enabled else 'disabled'}")

    def _on_comparison_sources_changed(self, selected_sources: list):
        """Handle comparison source selection change."""
        logger.debug(f"Grade-Tonnage: Comparison sources changed: {selected_sources}")

        # Populate properties from selected sources
        if selected_sources:
            self._populate_comparison_properties(selected_sources)

        # Enable run button if at least 2 sources selected
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(len(selected_sources) >= 2)

    def _populate_comparison_properties(self, selected_keys: List[str]):
        """Populate grade column dropdown with properties from selected sources."""
        import pandas as pd

        all_properties = set()

        for source_key in selected_keys:
            data = None
            if source_key == 'block_model':
                data = self._stored_block_model
            elif source_key == 'classified_block_model':
                data = self._stored_classified_block_model
            elif source_key in self._block_model_sources:
                data = self._block_model_sources[source_key].get('df')

            if data is not None:
                logger.debug(f"Grade-Tonnage: Source {source_key} type: {type(data).__name__}")
                # Check if it's a DataFrame
                if isinstance(data, pd.DataFrame):
                    for col in data.columns:
                        if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG'):
                            try:
                                if pd.api.types.is_numeric_dtype(data[col]):
                                    all_properties.add(col)
                            except:
                                all_properties.add(col)
                    logger.debug(f"Grade-Tonnage: DataFrame columns: {list(data.columns)[:10]}")
                else:
                    # BlockModel class
                    if hasattr(data, 'properties') and data.properties:
                        for prop in data.properties.keys():
                            all_properties.add(prop)
                        logger.debug(f"Grade-Tonnage: BlockModel properties: {list(data.properties.keys())[:5]}")
                    if hasattr(data, 'to_dataframe'):
                        try:
                            df = data.to_dataframe()
                            for col in df.columns:
                                if col.upper() not in ('X', 'Y', 'Z', 'DX', 'DY', 'DZ'):
                                    if pd.api.types.is_numeric_dtype(df[col]):
                                        all_properties.add(col)
                            logger.debug(f"Grade-Tonnage: to_dataframe columns: {list(df.columns)[:10]}")
                        except Exception as e:
                            logger.debug(f"Grade-Tonnage: to_dataframe failed: {e}")

        # Update grade column dropdown
        if hasattr(self, 'grade_combo'):
            current = self.grade_combo.currentText()
            self.grade_combo.blockSignals(True)
            self.grade_combo.clear()

            sorted_props = sorted(all_properties)
            self.grade_combo.addItems(sorted_props)

            if current and current in sorted_props:
                self.grade_combo.setCurrentText(current)
            else:
                for prop in sorted_props:
                    if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                        self.grade_combo.setCurrentText(prop)
                        break

            self.grade_combo.blockSignals(False)
            logger.info(f"Grade-Tonnage: Populated {len(sorted_props)} properties for comparison")

    def _update_comparison_sources(self):
        """Update the comparison source list with available block model sources."""
        sources = {}

        # Add regular block model
        if self._stored_block_model is not None:
            sources['block_model'] = {
                'display_name': 'Block Model',
                'block_count': len(self._stored_block_model),
                'df': self._stored_block_model
            }

        # Add classified block model
        if self._stored_classified_block_model is not None:
            sources['classified_block_model'] = {
                'display_name': 'Classified Block Model',
                'block_count': len(self._stored_classified_block_model),
                'df': self._stored_classified_block_model
            }

        # Add individual SGSIM sources
        for source_key, source_info in self._block_model_sources.items():
            if source_key.startswith('sgsim_'):
                df = source_info.get('df')
                if df is not None:
                    sources[source_key] = {
                        'display_name': source_info.get('display_name', source_key),
                        'block_count': len(df),
                        'df': df
                    }

        self._source_selection_widget.update_sources(sources)
        logger.debug(f"Grade-Tonnage: Updated comparison sources with {len(sources)} sources")

    def _get_data_for_source(self, source_key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame for a given source key."""
        if source_key == 'block_model':
            return self._stored_block_model
        elif source_key == 'classified_block_model':
            return self._stored_classified_block_model
        elif source_key == 'sgsim':
            return self._stored_sgsim_df
        elif source_key.startswith('sgsim_') and source_key in self._block_model_sources:
            return self._block_model_sources[source_key].get('df')
        return None

    def _run_comparison_analysis(self):
        """Run grade-tonnage analysis for multiple sources and create comparison plots."""
        selected_sources = self._source_selection_widget.get_selected_sources()

        if len(selected_sources) < 2:
            QMessageBox.warning(
                self, "Comparison Mode",
                "Please select at least 2 sources to compare."
            )
            return

        if self.is_analyzing:
            QMessageBox.warning(self, "Analysis in Progress", "Please wait for current analysis to complete.")
            return

        try:
            # Get column configuration
            grade_col = self.grade_col.currentText()
            tonnage_col = self.tonnage_col.currentText()

            if not grade_col:
                QMessageBox.warning(self, "Validation Error", "Please select a grade column.")
                return

            # Update UI state
            self.is_analyzing = True
            self.run_analysis_btn.setEnabled(False)
            self.run_analysis_btn.setText("🔄 Running Comparison...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Clear previous results
            self._comparison_results.clear()
            self._clear_results_display()

            # Process each source
            cutoff_range = np.linspace(
                self.cutoff_min.value(),
                self.cutoff_max.value(),
                self.cutoff_steps.value()
            )

            for i, source_key in enumerate(selected_sources):
                df = self._get_data_for_source(source_key)
                if df is None:
                    logger.warning(f"Grade-Tonnage: No data for source {source_key}")
                    continue

                # Get display name
                display_name = source_key
                if source_key in self._block_model_sources:
                    display_name = self._block_model_sources[source_key].get('display_name', source_key)
                elif source_key == 'block_model':
                    display_name = 'Block Model'
                elif source_key == 'classified_block_model':
                    display_name = 'Classified Block Model'

                # Calculate grade-tonnage curve for this source
                try:
                    result = self._calculate_gt_curve_for_source(df, grade_col, tonnage_col, cutoff_range)
                    if result is not None:
                        self._comparison_results[source_key] = {
                            'display_name': display_name,
                            'cutoffs': cutoff_range,
                            'tonnage': result['tonnage'],
                            'grade': result['grade'],
                            'metal': result['metal'],
                            'total_tonnage': result['total_tonnage'],
                            'df': df
                        }
                except Exception as e:
                    logger.error(f"Grade-Tonnage: Error calculating GT for {source_key}: {e}")

                # Update progress
                progress = int((i + 1) / len(selected_sources) * 100)
                self.progress_bar.setValue(progress)

            # Plot comparison results
            if len(self._comparison_results) >= 2:
                self._plot_comparison_gt_curves()
                self._update_comparison_stats_table()
                self.status_text.setPlainText(
                    f"✅ Comparison complete! Analyzed {len(self._comparison_results)} sources."
                )
            else:
                self.status_text.setPlainText("⚠️ Not enough valid sources for comparison.")

        except Exception as e:
            logger.exception(f"Grade-Tonnage: Comparison analysis error: {e}")
            QMessageBox.critical(self, "Error", f"Comparison analysis failed: {str(e)}")
        finally:
            self.is_analyzing = False
            self.run_analysis_btn.setEnabled(True)
            self.run_analysis_btn.setText("🚀 Run Geostatistical Analysis")
            self.progress_bar.setVisible(False)

    def _calculate_gt_curve_for_source(
        self,
        df: pd.DataFrame,
        grade_col: str,
        tonnage_col: str,
        cutoff_range: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Calculate grade-tonnage curve for a single source."""
        try:
            # Flexible grade column matching
            actual_grade_col = grade_col
            if grade_col not in df.columns:
                found_match = False

                # Strategy 1: Column contains grade_col
                for col in df.columns:
                    if grade_col.upper() in col.upper():
                        actual_grade_col = col
                        found_match = True
                        logger.debug(f"Grade-Tonnage: Matched {grade_col} -> {col}")
                        break

                # Strategy 2: grade_col contains column name
                if not found_match:
                    for col in df.columns:
                        if col.upper() in grade_col.upper() and col.upper() not in ('X', 'Y', 'Z'):
                            actual_grade_col = col
                            found_match = True
                            logger.debug(f"Grade-Tonnage: Matched {grade_col} -> {col}")
                            break

                # Strategy 3: Base name match
                if not found_match:
                    base_name = grade_col.split('_')[0] if '_' in grade_col else grade_col
                    for col in df.columns:
                        col_base = col.split('_')[0] if '_' in col else col
                        if base_name.upper() == col_base.upper() and col.upper() not in ('X', 'Y', 'Z'):
                            actual_grade_col = col
                            found_match = True
                            logger.debug(f"Grade-Tonnage: Matched {grade_col} -> {col}")
                            break

                # Strategy 4: Any grade-like property
                if not found_match:
                    for col in df.columns:
                        if any(k in col.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and col.upper() not in ('X', 'Y', 'Z'):
                            actual_grade_col = col
                            found_match = True
                            logger.debug(f"Grade-Tonnage: Using grade-like property {col}")
                            break

                if not found_match:
                    logger.warning(f"Grade column '{grade_col}' not found in DataFrame. Available: {list(df.columns)[:10]}")
                    return None

            grades = df[actual_grade_col].values
            tonnages = df[tonnage_col].values if tonnage_col in df.columns else np.ones(len(df))

            # Calculate GT curve
            result_tonnage = []
            result_grade = []
            result_metal = []
            total_tonnage = np.sum(tonnages)

            for cutoff in cutoff_range:
                mask = grades >= cutoff
                above_cutoff_tonnage = np.sum(tonnages[mask])
                if above_cutoff_tonnage > 0:
                    above_cutoff_grade = np.average(grades[mask], weights=tonnages[mask])
                else:
                    above_cutoff_grade = 0.0

                result_tonnage.append(above_cutoff_tonnage)
                result_grade.append(above_cutoff_grade)
                result_metal.append(above_cutoff_tonnage * above_cutoff_grade / 100.0)  # Assuming % grade

            return {
                'tonnage': np.array(result_tonnage),
                'grade': np.array(result_grade),
                'metal': np.array(result_metal),
                'total_tonnage': total_tonnage
            }

        except Exception as e:
            logger.error(f"Error calculating GT curve: {e}")
            return None

    def _plot_comparison_gt_curves(self):
        """Plot overlaid grade-tonnage curves for comparison."""
        if not self._comparison_results:
            return

        try:
            # Get or create figure
            if not hasattr(self, 'curve_canvas') or self.curve_canvas is None:
                from matplotlib.figure import Figure
                fig = Figure(figsize=(10, 8), dpi=100)
                self.curve_canvas = FigureCanvas(fig)
            else:
                fig = self.curve_canvas.figure
                fig.clear()

            # Create dual-axis plot
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()

            source_names = []
            for i, (source_key, result) in enumerate(self._comparison_results.items()):
                style = ComparisonColors.get_style(i)
                display_name = result['display_name']
                source_names.append(display_name)

                cutoffs = result['cutoffs']
                tonnage = result['tonnage']
                grade = result['grade']

                # Normalize tonnage to percentage of total
                total = result['total_tonnage']
                if total > 0:
                    tonnage_pct = tonnage / total * 100
                else:
                    tonnage_pct = tonnage

                # Plot tonnage (left axis) - solid lines
                ax1.plot(
                    cutoffs, tonnage_pct,
                    color=style['color'],
                    linestyle='-',
                    linewidth=2,
                    label=f'{display_name} (Tonnage)',
                    alpha=style['alpha']
                )

                # Plot grade (right axis) - dashed lines
                ax2.plot(
                    cutoffs, grade,
                    color=style['color'],
                    linestyle='--',
                    linewidth=2,
                    label=f'{display_name} (Grade)',
                    alpha=style['alpha']
                )

            # Styling with theme colors
            colors = get_theme_colors()
            ax1.set_xlabel('Cutoff Grade', fontsize=11, color=colors.TEXT_PRIMARY)
            ax1.set_ylabel('Tonnage (% of Total)', fontsize=11, color=colors.TEXT_PRIMARY)
            ax2.set_ylabel('Average Grade', fontsize=11, color=colors.TEXT_PRIMARY)

            ax1.set_facecolor(colors.CARD_BG)
            fig.set_facecolor(colors.PANEL_BG)

            ax1.tick_params(colors=colors.TEXT_PRIMARY)
            ax2.tick_params(colors=colors.TEXT_PRIMARY)
            for spine in ax1.spines.values():
                spine.set_color(colors.BORDER)
            ax2.spines['right'].set_color(colors.BORDER)

            ax1.grid(True, alpha=0.3, color=colors.BORDER)

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines1 + lines2, labels1 + labels2,
                loc='upper right',
                frameon=True,
                facecolor=colors.ELEVATED_BG,
                edgecolor=colors.BORDER,
                fontsize=9,
                labelcolor=colors.TEXT_PRIMARY
            )

            ax1.set_title(
                'Grade-Tonnage Comparison',
                fontsize=14,
                fontweight='bold',
                color=f'{ModernColors.TEXT_PRIMARY}',
                pad=15
            )

            fig.tight_layout()
            self.curve_canvas.draw()

            # Show canvas
            if hasattr(self, 'curve_placeholder'):
                self.curve_placeholder.hide()
            self.curve_canvas.show()

            # Enable popup button
            if hasattr(self, 'popup_curve_btn'):
                self.popup_curve_btn.setEnabled(True)

            logger.info(f"Grade-Tonnage: Plotted comparison curves for {len(self._comparison_results)} sources")

        except Exception as e:
            logger.error(f"Error plotting comparison GT curves: {e}", exc_info=True)

    def _update_comparison_stats_table(self):
        """Update statistics display with comparison table."""
        if not self._comparison_results:
            return

        try:
            # Build comparison summary text
            summary_lines = [
                "=" * 60,
                "GRADE-TONNAGE COMPARISON SUMMARY",
                "=" * 60,
                ""
            ]

            # Header
            header = f"{'Source':<30} {'Total Tonnage':>15} {'Avg Grade':>12}"
            summary_lines.append(header)
            summary_lines.append("-" * 60)

            for source_key, result in self._comparison_results.items():
                display_name = result['display_name'][:28]
                total_tonnage = result['total_tonnage']
                # Average grade at zero cutoff
                avg_grade = result['grade'][0] if len(result['grade']) > 0 else 0

                line = f"{display_name:<30} {total_tonnage:>15,.0f} {avg_grade:>12.3f}"
                summary_lines.append(line)

            summary_lines.append("-" * 60)
            summary_lines.append("")
            summary_lines.append("Note: Tonnage curves shown as % of each source's total.")
            summary_lines.append("Solid lines = Tonnage, Dashed lines = Grade")

            self.status_text.setPlainText("\n".join(summary_lines))

        except Exception as e:
            logger.error(f"Error updating comparison stats: {e}")

    def _run_geostats_analysis(self):
        """Run the comprehensive geostatistical grade-tonnage analysis."""
        if self._block_model_data is None:
            QMessageBox.warning(self, "No Data", "Please load block model data first.")
            return

        if self.is_analyzing:
            QMessageBox.warning(self, "Analysis in Progress", "Please wait for current analysis to complete.")
            return

        try:
            # Validate inputs
            if not self._validate_analysis_inputs():
                return

            # Update UI state
            self.is_analyzing = True
            self.run_analysis_btn.setEnabled(False)
            self.run_analysis_btn.setText("🔄 Analyzing...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Disable clear and export buttons during analysis
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setEnabled(False)
            if hasattr(self, 'export_btn'):
                self.export_btn.setEnabled(False)

            # Clear previous results
            self._clear_results_display()

            # Update status
            self.status_text.setPlainText("Starting geostatistical analysis...")
            self.analysisProgress.emit("Initializing analysis parameters...")

            # Collect analysis parameters
            config = self._collect_analysis_config()
            cutoff_range = np.linspace(
                self.cutoff_min.value(),
                self.cutoff_max.value(),
                self.cutoff_steps.value()
            )

            # Collect column configuration
            column_config = {
                'grade': self.grade_col.currentText(),
                'tonnage': self.tonnage_col.currentText(),
                'x': self.x_col.currentText(),
                'y': self.y_col.currentText(),
                'z': self.z_col.currentText()
            }

            # Collect advanced analysis configuration
            advanced_config = {
                'enable_domain_analysis': hasattr(self, 'enable_domain_analysis') and self.enable_domain_analysis.isChecked(),
                'domain_column': self.domain_column.currentText() if hasattr(self, 'domain_column') else None,
                'enable_classification_analysis': hasattr(self, 'enable_classification_analysis') and self.enable_classification_analysis.isChecked(),
                'classification_column': self.classification_column.currentText() if hasattr(self, 'classification_column') else None,
            }

            # Start analysis in background thread
            self.analysis_thread = AnalysisWorker(
                block_model_data=self._block_model_data,
                config=config,
                cutoff_range=cutoff_range,
                gt_engine=self.gt_engine,
                sensitivity_engine=self.sensitivity_engine,
                column_config=column_config,
                advanced_config=advanced_config
            )

            self.analysis_thread.progress.connect(self._on_analysis_progress)
            self.analysis_thread.finished.connect(self._on_analysis_complete)
            self.analysis_thread.advanced_finished.connect(self._on_advanced_analysis_complete)
            self.analysis_thread.error.connect(self._on_analysis_error)
            self.analysis_thread.start()

        except Exception as e:
            self._handle_analysis_error(f"Failed to start analysis: {str(e)}")
            logger.exception("Analysis startup error")

    def _validate_analysis_inputs(self) -> bool:
        """Validate analysis input parameters."""
        # Check required columns
        grade_col = self.grade_col.currentText()
        tonnage_col = self.tonnage_col.currentText()
        x_col = self.x_col.currentText()
        y_col = self.y_col.currentText()
        z_col = self.z_col.currentText()

        if not grade_col:
            QMessageBox.warning(self, "Validation Error", "Please select a grade column.")
            return False

        if not tonnage_col:
            QMessageBox.warning(self, "Validation Error", "Please select a tonnage column.")
            return False

        if not all([x_col, y_col, z_col]):
            QMessageBox.warning(self, "Validation Error", "Please select coordinate columns (X, Y, Z).")
            return False

        # Validate data
        validation_errors = validate_grade_tonnage_data(
            self._block_model_data,
            grade_col=grade_col,
            tonnage_col=tonnage_col
        )

        if validation_errors:
            error_msg = "Data validation issues found:\n\n" + "\n".join(f"• {err}" for err in validation_errors)
            error_msg += "\n\nDo you want to continue anyway?"
            reply = QMessageBox.question(
                self, "Data Validation Warning",
                error_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False

        return True

    def _collect_analysis_config(self) -> GeostatsGradeTonnageConfig:
        """Collect analysis configuration from UI controls."""
        # Economic parameters
        economic_params = {
            "metal_price": self.metal_price.value(),
            "mining_cost": self.mining_cost.value(),
            "processing_cost": self.processing_cost.value(),
            "recovery": self.recovery.value(),
            "discount_rate": self.discount_rate.value(),
            "admin_cost": 0.0,  # Could add UI control later
            "transport_cost": 0.0,  # Could add UI control later
            "dilution": 0.0,
            "mining_loss": 0.0
        }

        # Determine data mode from UI selection
        data_mode = DataMode.COMPOSITES if self.data_mode_combo.currentIndex() == 1 else DataMode.BLOCK_MODEL
        
        # Determine grade weighting method
        grade_weighting = (
            GradeWeightingMethod.TONNAGE_WEIGHTED 
            if self.grade_weighting_combo.currentIndex() == 0 
            else GradeWeightingMethod.EQUAL_WEIGHT
        )

        # Geostatistics configuration
        config = GeostatsGradeTonnageConfig(
            data_mode=data_mode,
            grade_weighting=grade_weighting,
            decluster_cell_size=self.cell_size.value(),
            decluster_method=DeclusteringMethod.CELL_DECLUSTERING,
            confidence_method=ConfidenceIntervalMethod.VARIANCE_ESTIMATION,
            economic_params=economic_params
        )
        
        logger.info(f"Analysis config: data_mode={data_mode.value}, grade_weighting={grade_weighting.value}, declustering={'enabled' if data_mode == DataMode.COMPOSITES else 'disabled'}")

        return config

    def _collect_mine_economics_config(self) -> MineEconomicsConfig:
        """
        Collect multi-period mine economics configuration from UI controls.
        
        This generates a full MineEconomicsConfig for audit-grade DCF analysis.
        Only used when advanced economics mode is enabled.
        """
        econ_params = EconomicParameters(
            metal_price=self.metal_price.value(),
            mining_cost_ore=self.mining_cost.value(),
            mining_cost_waste=self.mining_cost.value() * 0.5,  # Assume waste is cheaper
            processing_cost=self.processing_cost.value(),
            admin_cost=5.0,  # Default admin cost
            recovery_rate=self.recovery.value(),
            dilution_factor=self.dilution.value(),
            mining_loss_factor=self.mining_loss.value(),
            discount_rate=self.discount_rate.value()
        )
        
        capacity = MineCapacity(
            annual_ore_capacity=self.annual_capacity.value(),
            ramp_up_years=self.ramp_up_years.value()
        )
        
        capex = CapitalExpenditure(
            initial_capex=self.initial_capex.value(),
            sustaining_capex_per_tonne=self.sustaining_capex.value(),
            closure_cost=self.closure_cost.value()
        )
        
        tax_params = TaxParameters(
            corporate_tax_rate=self.tax_rate.value(),
            royalty_rate=self.royalty_rate.value()
        )
        
        config = MineEconomicsConfig(
            economic_params=econ_params,
            capacity=capacity,
            capex=capex,
            tax=tax_params
        )
        
        logger.info(f"Mine economics config: capacity={capacity.annual_ore_capacity:,.0f} t/year, "
                   f"initial_capex=${capex.initial_capex:,.0f}")
        
        return config

    def is_multi_period_enabled(self) -> bool:
        """Check if multi-period economics mode is enabled."""
        return hasattr(self, 'advanced_economics_group') and self.advanced_economics_group.isChecked()

    def _on_analysis_progress(self, message: str):
        """Handle analysis progress updates."""
        self.status_text.setPlainText(message)
        self.analysisProgress.emit(message)

        # Update progress bar based on message keywords
        if "Initializing" in message:
            self.progress_bar.setValue(10)
        elif "declustering" in message.lower():
            self.progress_bar.setValue(30)
        elif "grade-tonnage" in message.lower():
            self.progress_bar.setValue(60)
        elif "sensitivity" in message.lower():
            self.progress_bar.setValue(80)
        elif "visualization" in message.lower():
            self.progress_bar.setValue(95)

    def _on_analysis_complete(self, grade_tonnage_curve, sensitivity_analysis):
        """Handle analysis completion."""
        try:
            # Store results
            self.grade_tonnage_curve = grade_tonnage_curve
            self.sensitivity_analysis = sensitivity_analysis

            # Update UI
            self.is_analyzing = False
            self.run_analysis_btn.setEnabled(True)
            self.run_analysis_btn.setText("🚀 Run Geostatistical Analysis")
            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)
            self.export_btn.setEnabled(True)
            
            # Enable clear button (results are now available)
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setEnabled(True)
            
            # Enable popup buttons
            self.popup_curve_btn.setEnabled(True)
            self.popup_sensitivity_btn.setEnabled(True)
            self.popup_dashboard_btn.setEnabled(True)
            self.popup_all_btn.setEnabled(True)

            # Update status
            optimal_cutoff = sensitivity_analysis.optimal_cutoff
            self.status_text.setPlainText(
                f"✅ Analysis complete! Optimal cutoff grade: {optimal_cutoff:.2f}\n"
                f"Total tonnage: {grade_tonnage_curve.global_statistics['total_tonnage']:,.0f} tonnes\n"
                f"Average grade: {grade_tonnage_curve.global_statistics['grade_statistics']['mean']:.2f}"
            )

            # Update visualizations
            self._update_visualizations()

            # Emit completion signal
            self.analysisCompleted.emit(grade_tonnage_curve, sensitivity_analysis)

            logger.info("Geostatistical grade-tonnage analysis completed successfully")

        except Exception as e:
            self._handle_analysis_error(f"Error updating results: {str(e)}")
            logger.exception("Analysis completion error")

    def _on_analysis_error(self, error_message: str):
        """Handle analysis errors."""
        self._handle_analysis_error(error_message)

    def _on_advanced_analysis_complete(self, domain_result, classification_result):
        """Handle advanced analysis (domain/classification) completion."""
        try:
            # Update domain analysis tab
            if domain_result:
                self._update_domain_analysis_display(domain_result)
            
            # Update classification analysis tab
            if classification_result:
                self._update_classification_analysis_display(classification_result)
            
            logger.info("Advanced analysis results updated")
            
        except Exception as e:
            logger.error(f"Error updating advanced analysis display: {e}", exc_info=True)

    def _update_domain_analysis_display(self, result: DomainGTResult):
        """Update the domain analysis tab with results."""
        try:
            # Update summary table
            self.domain_summary_table.setRowCount(len(result.domains))
            
            for i, domain in enumerate(result.domains):
                stats = result.domain_statistics.get(domain, {})
                self.domain_summary_table.setItem(i, 0, QTableWidgetItem(str(domain)))
                self.domain_summary_table.setItem(i, 1, QTableWidgetItem(f"{stats.get('total_tonnage', 0):,.0f}"))
                self.domain_summary_table.setItem(i, 2, QTableWidgetItem(f"{stats.get('mean_grade', 0):.3f}"))
                self.domain_summary_table.setItem(i, 3, QTableWidgetItem(f"{stats.get('total_metal', 0):,.0f}"))
            
            self.domain_summary_table.setVisible(True)
            self.domain_placeholder.hide()
            
            # Update text summary
            summary_text = f"""
DOMAIN-WISE GRADE-TONNAGE ANALYSIS
{'='*50}

Total Domains: {len(result.domains)}

DOMAIN BREAKDOWN:
"""
            for domain in result.domains:
                stats = result.domain_statistics.get(domain, {})
                summary_text += f"""
• {domain}:
  - Blocks: {stats.get('block_count', 0):,}
  - Tonnage: {stats.get('total_tonnage', 0):,.0f} t
  - Mean Grade: {stats.get('mean_grade', 0):.3f}
  - Metal Content: {stats.get('total_metal', 0):,.0f}
"""
            
            self.domain_results_text.setPlainText(summary_text)
            
            logger.info(f"Domain analysis display updated: {len(result.domains)} domains")
            
        except Exception as e:
            logger.error(f"Error updating domain analysis display: {e}", exc_info=True)

    def _update_classification_analysis_display(self, result: ClassificationGTResult):
        """Update the classification analysis tab with results."""
        try:
            stats = result.category_statistics
            metadata = result.metadata
            
            # Update summary labels
            measured_stats = stats.get('measured', {})
            indicated_stats = stats.get('indicated', {})
            inferred_stats = stats.get('inferred', {})
            
            self.measured_summary.setText(
                f"{measured_stats.get('total_tonnage', 0):,.0f} t @ {measured_stats.get('mean_grade', 0):.3f}"
            )
            self.indicated_summary.setText(
                f"{indicated_stats.get('total_tonnage', 0):,.0f} t @ {indicated_stats.get('mean_grade', 0):.3f}"
            )
            self.inferred_summary.setText(
                f"{inferred_stats.get('total_tonnage', 0):,.0f} t @ {inferred_stats.get('mean_grade', 0):.3f}"
            )
            
            total_tonnage = (
                measured_stats.get('total_tonnage', 0) +
                indicated_stats.get('total_tonnage', 0) +
                inferred_stats.get('total_tonnage', 0)
            )
            self.total_resource_summary.setText(f"{total_tonnage:,.0f} t")
            
            self.classification_placeholder.hide()
            
            # Update text summary
            summary_text = f"""
RESOURCE CLASSIFICATION ANALYSIS (JORC/SAMREC)
{'='*50}

MEASURED RESOURCES:
• Blocks: {measured_stats.get('block_count', 0):,}
• Tonnage: {measured_stats.get('total_tonnage', 0):,.0f} t
• Mean Grade: {measured_stats.get('mean_grade', 0):.4f}
• Metal Content: {measured_stats.get('total_metal', 0):,.0f}

INDICATED RESOURCES:
• Blocks: {indicated_stats.get('block_count', 0):,}
• Tonnage: {indicated_stats.get('total_tonnage', 0):,.0f} t
• Mean Grade: {indicated_stats.get('mean_grade', 0):.4f}
• Metal Content: {indicated_stats.get('total_metal', 0):,.0f}

INFERRED RESOURCES:
• Blocks: {inferred_stats.get('block_count', 0):,}
• Tonnage: {inferred_stats.get('total_tonnage', 0):,.0f} t
• Mean Grade: {inferred_stats.get('mean_grade', 0):.4f}
• Metal Content: {inferred_stats.get('total_metal', 0):,.0f}

TOTAL MINERAL RESOURCE:
• Total Tonnage: {total_tonnage:,.0f} t
• Total Blocks: {metadata.get('total_blocks', 0):,}

Note: Mineral Resources are reported in accordance with JORC 2012 / SAMREC guidelines.
Resource classification is based on drillhole spacing and data confidence.
"""
            
            self.classification_results_text.setPlainText(summary_text)
            
            logger.info("Classification analysis display updated")
            
        except Exception as e:
            logger.error(f"Error updating classification analysis display: {e}", exc_info=True)

    def _handle_analysis_error(self, error_message: str):
        """Handle analysis errors and reset UI state."""
        self.is_analyzing = False
        self.run_analysis_btn.setEnabled(True)
        self.run_analysis_btn.setText("🚀 Run Geostatistical Analysis")
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(False)

        self.status_text.setPlainText(f"❌ Analysis failed: {error_message}")
        self.analysisError.emit(error_message)

        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n\n{error_message}")
        logger.error(f"Analysis error: {error_message}")

    def _clear_results_display(self):
        """Clear all result displays."""
        # Clear tables
        if hasattr(self, 'curve_table'):
            self.curve_table.setRowCount(0)
            self.curve_table.setVisible(False)

        # Clear canvases
        for attr in ['curve_canvas', 'sensitivity_canvas', 'dashboard_canvas']:
            if hasattr(self, attr):
                canvas = getattr(self, attr)
                if canvas is not None:
                    canvas.hide()

        # Reset placeholders
        for attr in ['curve_placeholder', 'sensitivity_placeholder', 'dashboard_placeholder']:
            if hasattr(self, attr):
                getattr(self, attr).show()

        # Clear optimal display
        if hasattr(self, 'optimal_display'):
            self.optimal_display.setVisible(False)

    def _clear_all_results(self):
        """
        Clear all analysis results, tables, visualizations, and statistics.
        
        This method provides a comprehensive reset of all analysis outputs
        while preserving the loaded data and configuration settings.
        """
        try:
            # Clear stored results
            self.grade_tonnage_curve = None
            self.sensitivity_analysis = None
            
            # Clear result displays
            self._clear_results_display()
            
            # Clear statistics text
            if hasattr(self, 'stats_text'):
                self.stats_text.setPlainText("No analysis results available.")
            
            # Clear validation text
            if hasattr(self, 'validation_text'):
                self.validation_text.setPlainText("No data loaded for validation.")
            
            # Clear advanced analysis results
            if hasattr(self, 'sgs_results_text'):
                self.sgs_results_text.setPlainText("No SGS uncertainty analysis results available.")
            
            if hasattr(self, 'domain_results_text'):
                self.domain_results_text.setPlainText("No domain-wise analysis results available.")
            
            if hasattr(self, 'classification_results_text'):
                self.classification_results_text.setPlainText("No classification analysis results available.")
            
            # Clear status text (but keep data loaded message if data exists)
            if self._block_model_data is not None:
                self.status_text.setPlainText(
                    f"✅ Block model loaded: {len(self._block_model_data):,} records\n"
                    f"Analysis results cleared. Ready for new analysis."
                )
            else:
                self.status_text.setPlainText("Analysis results cleared. Load block model data to begin.")
            
            # Disable export and clear buttons
            if hasattr(self, 'export_btn'):
                self.export_btn.setEnabled(False)
            if hasattr(self, 'clear_btn'):
                self.clear_btn.setEnabled(False)
            
            # Disable popup buttons
            for btn_name in ['popup_curve_btn', 'popup_sensitivity_btn', 'popup_dashboard_btn', 'popup_all_btn']:
                if hasattr(self, btn_name):
                    btn = getattr(self, btn_name)
                    btn.setEnabled(False)
            
            # Reset progress bar
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(0)
                self.progress_bar.setVisible(False)
            
            logger.info("All analysis results cleared")
            
            # Show confirmation
            QMessageBox.information(
                self,
                "Results Cleared",
                "All analysis results, tables, and visualizations have been cleared.\n\n"
                "Your data and configuration settings are preserved.\n"
                "You can run a new analysis with the current settings."
            )
            
        except Exception as e:
            logger.error(f"Error clearing results: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Clear Error",
                f"An error occurred while clearing results:\n\n{e}"
            )

    def _update_visualizations(self):
        """Update all visualization displays with analysis results."""
        if self.grade_tonnage_curve is None:
            return

        try:
            # Create plot configuration
            plot_config = PlotConfig(
                interactive=self.interactive_plots.isChecked(),
                color_scheme=ColorScheme(self.color_scheme.currentText().lower().replace(" ", "_")),
                show_confidence_intervals=self.show_confidence.isChecked()
            )
            self.viz_coordinator = MiningVisualizationCoordinator(plot_config)

            # Update grade-tonnage curve tab
            self._update_curve_visualization()

            # Update sensitivity analysis tab
            self._update_sensitivity_visualization()

            # Update dashboard tab
            self._update_dashboard_visualization()

            # Update statistics tab
            self._update_statistics_display()

        except Exception as e:
            logger.error(f"Error updating visualizations: {e}", exc_info=True)
            QMessageBox.warning(self, "Visualization Error",
                              f"Failed to update visualizations: {str(e)}")

    def _update_curve_visualization(self):
        """Update the grade-tonnage curve visualization."""
        try:
            # Create curve plot
            plot_obj = self.viz_coordinator.create_grade_tonnage_plot(
                self.grade_tonnage_curve,
                interactive=self.interactive_plots.isChecked(),
                title="Geostatistical Grade-Tonnage Curve"
            )

            # Update UI
            if isinstance(plot_obj, tuple):
                # Matplotlib
                fig, canvas = plot_obj
                if self.curve_canvas is not None:
                    self.curve_canvas.hide()
                self.curve_canvas = canvas
                self.curve_canvas.setParent(self.results_tabs.widget(0))
                self.results_tabs.widget(0).layout().replaceWidget(
                    self.curve_placeholder, self.curve_canvas)
                self.curve_placeholder.hide()
                self.curve_canvas.show()

                # Add NavigationToolbar for interactive zoom/pan
                if MATPLOTLIB_AVAILABLE and NavigationToolbar is not None:
                    # Remove old toolbar if exists
                    if self.curve_toolbar is not None:
                        self.curve_toolbar.hide()
                        self.curve_toolbar.deleteLater()

                    # Create new toolbar
                    self.curve_toolbar = NavigationToolbar(canvas, self)
                    # Insert at the beginning of toolbar layout (before export button)
                    self.curve_toolbar_layout.insertWidget(0, self.curve_toolbar)

                # Enable export button
                self.curve_export_btn.setEnabled(True)

            else:
                # Plotly - would need additional handling for Qt integration
                logger.warning("Plotly interactive plots not yet implemented for Qt integration")

            # Update curve data table
            self._populate_curve_table()

        except Exception as e:
            logger.error(f"Error updating curve visualization: {e}", exc_info=True)

    def _update_sensitivity_visualization(self):
        """Update the sensitivity analysis visualization."""
        if self.sensitivity_analysis is None:
            return

        try:
            # Create sensitivity plot
            plot_obj = self.viz_coordinator.create_sensitivity_plot(
                self.sensitivity_analysis,
                title="Cut-off Grade Sensitivity Analysis"
            )

            # Update optimal cutoff display
            optimal_cutoff = self.sensitivity_analysis.optimal_cutoff
            self.optimal_display.setText(
                f"🎯 Optimal Cutoff Grade: {optimal_cutoff:.2f}\n"
                f"Expected NPV: ${self.sensitivity_analysis.npv_by_cutoff.max():,.0f}"
            )
            self.optimal_display.show()

            # Update UI
            if isinstance(plot_obj, tuple):
                fig, canvas = plot_obj
                if self.sensitivity_canvas is not None:
                    self.sensitivity_canvas.hide()
                self.sensitivity_canvas = canvas
                self.sensitivity_canvas.setParent(self.results_tabs.widget(1))
                self.results_tabs.widget(1).layout().replaceWidget(
                    self.sensitivity_placeholder, self.sensitivity_canvas)
                self.sensitivity_placeholder.hide()
                self.sensitivity_canvas.show()

                # Add NavigationToolbar for interactive zoom/pan
                if MATPLOTLIB_AVAILABLE and NavigationToolbar is not None:
                    # Remove old toolbar if exists
                    if self.sensitivity_toolbar is not None:
                        self.sensitivity_toolbar.hide()
                        self.sensitivity_toolbar.deleteLater()

                    # Create new toolbar
                    self.sensitivity_toolbar = NavigationToolbar(canvas, self)
                    # Insert at the beginning of toolbar layout (before export button)
                    self.sensitivity_toolbar_layout.insertWidget(0, self.sensitivity_toolbar)

                # Enable export button
                self.sensitivity_export_btn.setEnabled(True)

        except Exception as e:
            logger.error(f"Error updating sensitivity visualization: {e}", exc_info=True)

    def _update_dashboard_visualization(self):
        """Update the dashboard visualization."""
        try:
            # Create dashboard plot
            plot_obj = self.viz_coordinator.create_dashboard(
                self.grade_tonnage_curve,
                self.sensitivity_analysis,
                interactive=self.interactive_plots.isChecked(),
                title="Geostatistical Mining Analysis Dashboard"
            )

            # Update UI
            if isinstance(plot_obj, tuple):
                fig, canvas = plot_obj
                if self.dashboard_canvas is not None:
                    self.dashboard_canvas.hide()
                self.dashboard_canvas = canvas
                self.dashboard_canvas.setParent(self.results_tabs.widget(2))
                self.results_tabs.widget(2).layout().replaceWidget(
                    self.dashboard_placeholder, self.dashboard_canvas)
                self.dashboard_placeholder.hide()
                self.dashboard_canvas.show()

        except Exception as e:
            logger.error(f"Error updating dashboard visualization: {e}", exc_info=True)

    def _update_statistics_display(self):
        """Update the statistics and summary display."""
        if self.grade_tonnage_curve is None:
            return

        try:
            # Generate comprehensive statistics report
            stats_report = self._generate_statistics_report()
            self.stats_text.setPlainText(stats_report)

            # Update validation results
            validation_report = self._generate_validation_report()
            self.validation_text.setPlainText(validation_report)

        except Exception as e:
            logger.error(f"Error updating statistics display: {e}", exc_info=True)

    def _open_plot_window(self, plot_type: str):
        """
        Open plot(s) in separate popup window(s) for better viewing.
        
        Args:
            plot_type: 'curve', 'sensitivity', 'dashboard', or 'all'
        """
        if self.grade_tonnage_curve is None:
            QMessageBox.warning(self, "No Data", "Please run analysis first.")
            return
        
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            
            # Create plot configuration
            plot_config = PlotConfig(
                interactive=False,
                color_scheme=ColorScheme(self.color_scheme.currentText().lower().replace(" ", "_")),
                show_confidence_intervals=self.show_confidence.isChecked(),
                figsize=(14, 10),  # Larger for popup windows
                dpi=100
            )
            viz = MiningVisualizationCoordinator(plot_config)
            
            if plot_type == "all":
                # Open all plots in separate windows
                self._open_plot_window("curve")
                self._open_plot_window("sensitivity")
                self._open_plot_window("dashboard")
                return
            
            # Create the appropriate plot
            if plot_type == "curve":
                fig, canvas = viz.create_grade_tonnage_plot(
                    self.grade_tonnage_curve,
                    interactive=False,
                    title="Geostatistical Grade-Tonnage Curve"
                )
                title = "Grade-Tonnage Curve"
                
            elif plot_type == "sensitivity":
                if self.sensitivity_analysis is None:
                    QMessageBox.warning(self, "No Data", "Sensitivity analysis not available.")
                    return
                fig, canvas = viz.create_sensitivity_plot(
                    self.sensitivity_analysis,
                    title="Cut-off Grade Sensitivity Analysis"
                )
                title = "Sensitivity Analysis"
                
            elif plot_type == "dashboard":
                if self.sensitivity_analysis is None:
                    QMessageBox.warning(self, "No Data", "Sensitivity analysis not available.")
                    return
                fig, canvas = viz.create_dashboard(
                    self.grade_tonnage_curve,
                    self.sensitivity_analysis,
                    interactive=False,
                    title="Mining Analysis Dashboard"
                )
                title = "Analysis Dashboard"
            else:
                return
            
            # Create popup dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"📊 {title}")
            dialog.setMinimumSize(1200, 900)
            dialog.resize(1400, 1000)
            
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            
            # Add canvas to dialog
            layout.addWidget(canvas)
            
            # Add save button
            btn_layout = QHBoxLayout()
            btn_layout.addStretch()
            
            save_btn = QPushButton("💾 Save as Image")
            save_btn.clicked.connect(lambda: self._save_popup_plot(fig, title))
            save_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4caf50;
                    color: white;
                    font-weight: bold;
                    padding: 8px 20px;
                    border-radius: 4px;
                }
                QPushButton:hover { background-color: #45a049; }
            """)
            btn_layout.addWidget(save_btn)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            btn_layout.addWidget(close_btn)
            
            layout.addLayout(btn_layout)
            
            # Show as non-modal so multiple windows can be open
            dialog.show()
            
            logger.info(f"Opened {plot_type} plot in separate window")
            
        except Exception as e:
            logger.error(f"Error opening plot window: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to open plot window:\n{str(e)}")

    def _save_popup_plot(self, fig, title: str):
        """Save the popup plot to file."""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Save {title}",
                f"{title.replace(' ', '_').lower()}.png",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )
            
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                QMessageBox.information(self, "Saved", f"Plot saved to:\n{filename}")
                logger.info(f"Plot saved to {filename}")
                
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save plot:\n{str(e)}")

    def _export_chart(self, chart_type: str):
        """Export a chart to PNG, SVG, or PDF.

        Args:
            chart_type: 'curve' or 'sensitivity'
        """
        # Get the appropriate canvas
        if chart_type == "curve":
            canvas = self.curve_canvas
            title = "Grade-Tonnage Curve"
        elif chart_type == "sensitivity":
            canvas = self.sensitivity_canvas
            title = "Sensitivity Analysis"
        else:
            return

        if canvas is None:
            QMessageBox.warning(self, "No Chart", "Please run analysis first to generate charts.")
            return

        try:
            filename, selected_filter = QFileDialog.getSaveFileName(
                self,
                f"Export {title}",
                f"{title.replace(' ', '_').lower()}.png",
                "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)"
            )

            if filename:
                # Get figure from canvas
                fig = canvas.figure

                # Determine format from filter or extension
                if filename.lower().endswith('.svg'):
                    fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                elif filename.lower().endswith('.pdf'):
                    fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                else:
                    fig.savefig(filename, format='png', dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')

                QMessageBox.information(self, "Exported", f"Chart exported successfully:\n{filename}")
                logger.info(f"Chart exported to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export chart:\n{str(e)}")
            logger.error(f"Chart export error: {e}", exc_info=True)

    def _populate_curve_table(self):
        """Populate the curve data table with results."""
        if self.grade_tonnage_curve is None:
            return

        try:
            points = self.grade_tonnage_curve.points
            self.curve_table.setRowCount(len(points))

            for i, point in enumerate(points):
                self.curve_table.setItem(i, 0, QTableWidgetItem(f"{point.cutoff_grade:.2f}"))
                self.curve_table.setItem(i, 1, QTableWidgetItem(f"{point.tonnage:,.0f}"))
                self.curve_table.setItem(i, 2, QTableWidgetItem(f"{point.avg_grade:.3f}"))
                self.curve_table.setItem(i, 3, QTableWidgetItem(f"{point.metal_quantity:,.0f}"))
                self.curve_table.setItem(i, 4, QTableWidgetItem(f"${point.net_value:,.0f}"))
                self.curve_table.setItem(i, 5, QTableWidgetItem(f"{point.cv_uncertainty_band[0]:,.0f}"))
                self.curve_table.setItem(i, 6, QTableWidgetItem(f"{point.cv_uncertainty_band[1]:,.0f}"))
                self.curve_table.setItem(i, 7, QTableWidgetItem(f"{point.cv_factor:.3f}"))
                self.curve_table.setItem(i, 8, QTableWidgetItem(f"{point.decluster_weight:.3f}"))

            self.curve_table.setVisible(True)

        except Exception as e:
            logger.error(f"Error populating curve table: {e}", exc_info=True)

    def _generate_statistics_report(self) -> str:
        """Generate comprehensive statistics report."""
        if self.grade_tonnage_curve is None:
            return "No analysis results available."

        stats = self.grade_tonnage_curve.global_statistics
        grade_stats = stats.get('grade_statistics', {})
        data_mode = stats.get('data_mode', 'block_model')
        grade_weighting = stats.get('grade_weighting', 'tonnage_weighted')
        declustered = stats.get('declustered', False)
        weighting_method = grade_stats.get('weighting_method', 'tonnage-weighted')

        report = f"""
GRADE-TONNAGE ANALYSIS REPORT
{'='*50}

DATA MODE: {data_mode.upper()}
{'• Block model input - no declustering applied' if data_mode == 'block_model' else '• Composite samples - cell declustering applied'}

DEPOSIT STATISTICS:
• Total Tonnage: {stats.get('total_tonnage', 0):,.0f} tonnes
• Total Metal: {stats.get('total_metal', 0):,.0f} units
• Mean Grade × Tonnage: {stats.get('total_metal', 0):,.0f} units
• Sample/Block Count: {stats.get('sample_count', 0):,}
• Declustering Applied: {'Yes' if declustered else 'No'}

GRADE DISTRIBUTION:
• Mean Grade ({weighting_method}): {grade_stats.get('mean', 0):.4f}
• Mean Grade (raw arithmetic): {grade_stats.get('mean_raw', grade_stats.get('mean', 0)):.4f}
• Mean Grade (tonnage-weighted): {grade_stats.get('mean_tonnage_weighted', grade_stats.get('mean', 0)):.4f}
• Median Grade: {grade_stats.get('median', 0):.3f}
• Standard Deviation: {grade_stats.get('std', 0):.3f}
• Coefficient of Variation: {grade_stats.get('cv', 0):.2%}
• Range: {grade_stats.get('min', 0):.3f} - {grade_stats.get('max', 0):.3f}
• Interquartile Range: {grade_stats.get('q25', 0):.3f} - {grade_stats.get('q75', 0):.3f}

ANALYSIS PARAMETERS:
• Data Mode: {data_mode.replace('_', ' ').title()}
• Grade Weighting: {grade_weighting.replace('_', '-').title()}
• Cutoff Range: {self.cutoff_min.value():.2f} - {self.cutoff_max.value():.2f}
• Cutoff Steps: {self.cutoff_steps.value()}
• Uncertainty Bands: {'Enabled (±CV heuristic)' if self.show_confidence.isChecked() else 'Disabled'}
"""
        if declustered:
            report += f"• Declustering Cell Size: {self.cell_size.value():.1f} m\n"
        
        report += """
NOTE: Uncertainty bands shown are heuristic ±CV bands on tonnage.
These are NOT formal geostatistical confidence intervals.
For proper uncertainty quantification, use SGS realisations.
"""

        if self.sensitivity_analysis:
            opt_cutoff = self.sensitivity_analysis.optimal_cutoff
            max_npv = self.sensitivity_analysis.npv_by_cutoff.max()

            report += f"""
OPTIMIZATION RESULTS:
• Optimal Cutoff Grade: {opt_cutoff:.2f}
• Maximum NPV: ${max_npv:,.0f}
• Optimization Method: {self.optimization_method.currentText()}

ECONOMIC PARAMETERS:
• Metal Price: ${self.metal_price.value():.2f}/unit
• Mining Cost: ${self.mining_cost.value():.2f}/tonne
• Processing Cost: ${self.processing_cost.value():.2f}/tonne
• Recovery Rate: {self.recovery.value():.1%}
• Discount Rate: {self.discount_rate.value():.1%}
"""

        # Add multi-period economics info if enabled
        if self.is_multi_period_enabled():
            report += f"""
MULTI-PERIOD ECONOMICS (Enabled):
• Annual Capacity: {self.annual_capacity.value():,.0f} t/year
• Ramp-up Period: {self.ramp_up_years.value()} years
• Initial CAPEX: ${self.initial_capex.value():,.0f}
• Sustaining CAPEX: ${self.sustaining_capex.value():.2f}/tonne
• Closure Cost: ${self.closure_cost.value():,.0f}
• Tax Rate: {self.tax_rate.value():.0%}
• Royalty Rate: {self.royalty_rate.value():.0%}
• Mining Dilution: {self.dilution.value():.0%}
• Mining Loss: {self.mining_loss.value():.0%}

NOTE: Multi-period DCF with capacity constraints provides
audit-grade NPV for JORC/SAMREC cutoff optimization claims.
"""

        return report

    def _generate_validation_report(self) -> str:
        """Generate data validation report."""
        if self._block_model_data is None:
            return "No data loaded for validation."

        grade_col = self.grade_col.currentText()
        tonnage_col = self.tonnage_col.currentText()

        validation_issues = validate_grade_tonnage_data(
            self._block_model_data, grade_col, tonnage_col
        )

        if not validation_issues:
            return "✅ Data validation passed - no issues found."

        report = "⚠️ DATA VALIDATION ISSUES FOUND:\n\n"
        for issue in validation_issues:
            report += f"• {issue}\n"

        report += "\nRecommendations:\n"
        report += "• Check for data entry errors\n"
        report += "• Verify coordinate system consistency\n"
        report += "• Ensure all required columns are populated\n"
        report += "• Consider data cleaning and outlier removal"

        return report

    def _export_results(self):
        """Export analysis results to various formats."""
        if self.grade_tonnage_curve is None:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return

        try:
            # Get export filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Analysis Results",
                "geostats_grade_tonnage_analysis",
                "CSV Files (*.csv);;PDF Report (*.pdf);;Excel Files (*.xlsx)"
            )

            if not filename:
                return

            # Export based on file extension
            if filename.endswith('.csv'):
                self._export_to_csv(filename)
            elif filename.endswith('.pdf'):
                self._export_to_pdf(filename)
            elif filename.endswith('.xlsx'):
                self._export_to_excel(filename)
            else:
                QMessageBox.warning(self, "Export Error", "Unsupported file format.")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export results: {str(e)}")
            logger.exception("Export error")

    def _export_to_csv(self, filename: str):
        """Export results to CSV format."""
        try:
            # Export grade-tonnage curve data
            curve_data = []
            for point in self.grade_tonnage_curve.points:
                curve_data.append({
                    'cutoff_grade': point.cutoff_grade,
                    'tonnage': point.tonnage,
                    'avg_grade': point.avg_grade,
                    'metal_quantity': point.metal_quantity,
                    'net_value': point.net_value,
                    'cv_uncertainty_lower': point.cv_uncertainty_band[0],
                    'cv_uncertainty_upper': point.cv_uncertainty_band[1],
                    'cv_factor': point.cv_factor,
                    'decluster_weight': point.decluster_weight
                })

            df_curve = pd.DataFrame(curve_data)
            df_curve.to_csv(filename.replace('.csv', '_curve.csv'), index=False)

            # Export sensitivity data if available
            if self.sensitivity_analysis:
                sensitivity_data = {
                    'cutoff_range': self.sensitivity_analysis.cutoff_range,
                    'npv_values': self.sensitivity_analysis.npv_by_cutoff
                }
                df_sensitivity = pd.DataFrame(sensitivity_data)
                df_sensitivity.to_csv(filename.replace('.csv', '_sensitivity.csv'), index=False)

            # Export advanced analysis results if available
            exported_files = [
                filename.replace('.csv', '_curve.csv'),
                filename.replace('.csv', '_sensitivity.csv')
            ]
            
            # Export domain analysis
            if hasattr(self, 'analysis_thread') and self.analysis_thread and self.analysis_thread.domain_result:
                domain_file = filename.replace('.csv', '_domains.csv')
                if export_domain_gt_to_csv(self.analysis_thread.domain_result, domain_file):
                    exported_files.append(domain_file)
            
            # Export classification analysis
            if hasattr(self, 'analysis_thread') and self.analysis_thread and self.analysis_thread.classification_result:
                class_file = filename.replace('.csv', '_classification.csv')
                if export_classification_gt_to_csv(self.analysis_thread.classification_result, class_file):
                    exported_files.append(class_file)

            QMessageBox.information(self, "Export Complete",
                                  f"Results exported to:\n" + "\n".join(exported_files))

        except Exception as e:
            raise Exception(f"CSV export failed: {str(e)}")

    def _export_to_pdf(self, filename: str):
        """Export comprehensive PDF report."""
        try:
            # Create publication-quality plots
            from ..mine_planning.cutoff.advanced_visualization import create_publication_quality_plot

            success = create_publication_quality_plot(
                self.grade_tonnage_curve,
                self.sensitivity_analysis,
                output_filename=filename.replace('.pdf', ''),
                config=PlotConfig(dpi=300, export_format='pdf')
            )

            if success:
                QMessageBox.information(self, "Export Complete",
                                      f"Publication-quality report exported to:\n{filename}")
            else:
                raise Exception("PDF generation failed")

        except Exception as e:
            raise Exception(f"PDF export failed: {str(e)}")

    def _export_to_excel(self, filename: str):
        """Export results to Excel format with multiple sheets."""
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Curve data
                curve_data = []
                for point in self.grade_tonnage_curve.points:
                    curve_data.append({
                        'Cutoff Grade': point.cutoff_grade,
                        'Tonnage (tonnes)': point.tonnage,
                        'Average Grade': point.avg_grade,
                        'Metal Quantity': point.metal_quantity,
                        'Net Value ($)': point.net_value,
                        'CV Uncertainty Lower': point.cv_uncertainty_band[0],
                        'CV Uncertainty Upper': point.cv_uncertainty_band[1],
                        'CV Factor': point.cv_factor,
                        'Decluster Weight': point.decluster_weight
                    })

                df_curve = pd.DataFrame(curve_data)
                df_curve.to_excel(writer, sheet_name='Grade_Tonnage_Curve', index=False)

                # Sensitivity data
                if self.sensitivity_analysis:
                    sensitivity_data = {
                        'Cutoff Grade': self.sensitivity_analysis.cutoff_range,
                        'NPV ($)': self.sensitivity_analysis.npv_by_cutoff
                    }
                    df_sensitivity = pd.DataFrame(sensitivity_data)
                    df_sensitivity.to_excel(writer, sheet_name='Sensitivity_Analysis', index=False)

                # Statistics
                stats_data = {
                    'Parameter': ['Total Tonnage', 'Average Grade', 'Optimal Cutoff', 'Maximum NPV'],
                    'Value': [
                        self.grade_tonnage_curve.global_statistics['total_tonnage'],
                        self.grade_tonnage_curve.global_statistics['grade_statistics']['mean'],
                        self.sensitivity_analysis.optimal_cutoff if self.sensitivity_analysis else 0,
                        self.sensitivity_analysis.npv_by_cutoff.max() if self.sensitivity_analysis else 0
                    ]
                }
                df_stats = pd.DataFrame(stats_data)
                df_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)

            QMessageBox.information(self, "Export Complete",
                                  f"Comprehensive Excel report exported to:\n{filename}")

        except Exception as e:
            raise Exception(f"Excel export failed: {str(e)}")

    # ============================================================================
    # DATA MANAGEMENT METHODS
    # ============================================================================

    def _on_bm_loaded(self, bm):
        """Handle block model loading and update UI accordingly."""
        if bm is None:
            logger.warning("Grade-Tonnage: _on_bm_loaded called with None")
            return

        try:
            if hasattr(bm, 'to_dataframe'):
                self._block_model_data = bm.to_dataframe()
                logger.info(f"Grade-Tonnage: Converted BlockModel to DataFrame, shape: {self._block_model_data.shape}")
            elif isinstance(bm, pd.DataFrame):
                self._block_model_data = bm
                logger.info(f"Grade-Tonnage: Received DataFrame directly, shape: {self._block_model_data.shape}")
            else:
                logger.warning(f"Grade-Tonnage: Unknown block model type: {type(bm)}")
                return

            if self._block_model_data is None or len(self._block_model_data) == 0:
                logger.warning("Grade-Tonnage: Block model data is empty")
                return

            # Update UI controls with available columns
            self._update_column_selections()

            # Enable analysis button
            self.run_analysis_btn.setEnabled(True)

            # Update status
            self.status_text.setPlainText(
                f"✅ Block model loaded successfully!\n"
                f"Shape: {self._block_model_data.shape[0]:,} rows × {self._block_model_data.shape[1]} columns\n"
                f"Ready for geostatistical analysis."
            )

            logger.info(f"Grade-Tonnage: Successfully loaded block model with {len(self._block_model_data)} records")

        except Exception as e:
            logger.error(f"Grade-Tonnage: Error in _on_bm_loaded: {e}", exc_info=True)
            self.status_text.setPlainText(f"❌ Error loading block model: {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Failed to load block model:\n\n{str(e)}")

    def _update_column_selections(self):
        """Update column selection combo boxes with available columns."""
        if self._block_model_data is None:
            return

        try:
            all_cols = self._block_model_data.columns.tolist()
            numeric_cols = self._block_model_data.select_dtypes(include=[np.number]).columns.tolist()

            # Update grade column (numeric columns only)
            self.grade_col.clear()
            self.grade_col.addItems(numeric_cols)

            # Update tonnage column (numeric columns only)
            self.tonnage_col.clear()
            self.tonnage_col.addItems(numeric_cols)

            # Update coordinate columns (all columns, but prefer numeric)
            for coord_combo in [self.x_col, self.y_col, self.z_col]:
                coord_combo.clear()
                coord_combo.addItems(numeric_cols + [c for c in all_cols if c not in numeric_cols])

            # Update domain column (for advanced analysis)
            if hasattr(self, 'domain_column'):
                self.domain_column.clear()
                # Look for domain-related columns
                domain_candidates = [c for c in all_cols if any(
                    term in c.upper() for term in ['DOMAIN', 'LITH', 'ROCK', 'GEOLOGY', 'ZONE', 'ORE_TYPE']
                )]
                self.domain_column.addItems(domain_candidates + [c for c in all_cols if c not in domain_candidates])

            # Update classification column (for advanced analysis)
            if hasattr(self, 'classification_column'):
                self.classification_column.clear()
                # Look for classification-related columns
                class_candidates = [c for c in all_cols if any(
                    term in c.upper() for term in ['CLASS', 'CATEGORY', 'RESOURCE', 'MEASURED', 'INDICATED', 'INFERRED', 'MII']
                )]
                self.classification_column.addItems(class_candidates + [c for c in all_cols if c not in class_candidates])

            # Auto-select common column names
            self._auto_select_columns()

            logger.info(f"Grade-Tonnage: Updated column selections with {len(numeric_cols)} numeric columns")

        except Exception as e:
            logger.error(f"Grade-Tonnage: Error updating column selections: {e}", exc_info=True)

    def _auto_select_columns(self):
        """Auto-select common column names for better UX."""
        if self._block_model_data is None:
            return

        all_cols = self._block_model_data.columns.tolist()

        # Auto-select grade column
        grade_candidates = [c for c in all_cols if any(term in c.upper() for term in ['GRADE', 'AU', 'CU', 'FE', 'MO'])]
        if grade_candidates:
            self.grade_col.setCurrentText(grade_candidates[0])

        # Auto-select tonnage column
        tonnage_candidates = [c for c in all_cols if any(term in c.upper() for term in ['TONNAGE', 'TONNES', 'MASS'])]
        if tonnage_candidates:
            self.tonnage_col.setCurrentText(tonnage_candidates[0])

        # Auto-select coordinate columns
        x_candidates = [c for c in all_cols if any(term in c.upper() for term in ['X', 'XCENTRE', 'XCENTER', 'XCENT'])]
        y_candidates = [c for c in all_cols if any(term in c.upper() for term in ['Y', 'YCENTRE', 'YCENTER', 'YCENT'])]
        z_candidates = [c for c in all_cols if any(term in c.upper() for term in ['Z', 'ZCENTRE', 'ZCENTER', 'ZCENT', 'ELEVATION'])]

        if x_candidates:
            self.x_col.setCurrentText(x_candidates[0])
        if y_candidates:
            self.y_col.setCurrentText(y_candidates[0])
        if z_candidates:
            self.z_col.setCurrentText(z_candidates[0])

        # Auto-detect cutoff range based on selected grade column
        self._auto_detect_cutoff_range()

    def _auto_detect_cutoff_range(self):
        """Auto-detect appropriate cutoff range based on the grade column data.

        Uses P5/P95 percentiles to determine a sensible cutoff range
        that spans the majority of the grade distribution.
        """
        if self._block_model_data is None:
            return

        grade_col = self.grade_col.currentText()
        if not grade_col or grade_col not in self._block_model_data.columns:
            return

        try:
            grades = self._block_model_data[grade_col].dropna()
            if len(grades) == 0:
                return

            # Calculate grade statistics
            grade_min = grades.min()
            grade_max = grades.max()
            p5 = grades.quantile(0.05)
            p95 = grades.quantile(0.95)

            # Determine cutoff range:
            # - Start at 80% of P5 (or 0 if that's negative)
            # - End at P95 (captures 95% of data)
            cutoff_min = max(0, p5 * 0.8)
            cutoff_max = p95

            # Ensure we have a meaningful range
            if cutoff_max <= cutoff_min:
                cutoff_max = grade_max
            if cutoff_max <= cutoff_min:
                cutoff_max = cutoff_min + 1.0

            # Round based on grade magnitude for cleaner UI
            if grade_max > 50:
                # High grade values (e.g., Fe% ~30-65): round to integers
                cutoff_min = np.floor(cutoff_min)
                cutoff_max = np.ceil(cutoff_max)
                decimals = 0
            elif grade_max > 10:
                # Medium grade values: round to 1 decimal
                cutoff_min = np.floor(cutoff_min * 10) / 10
                cutoff_max = np.ceil(cutoff_max * 10) / 10
                decimals = 1
            elif grade_max > 1:
                # Lower grade values (e.g., Cu% ~0.1-2): round to 1 decimal
                cutoff_min = np.floor(cutoff_min * 10) / 10
                cutoff_max = np.ceil(cutoff_max * 10) / 10
                decimals = 2
            else:
                # Very low grade values (e.g., Au g/t): keep 2 decimals
                decimals = 2

            # Update spinbox ranges and values
            self.cutoff_min.setRange(0.0, grade_max * 1.2)
            self.cutoff_max.setRange(0.0, grade_max * 1.2)
            self.cutoff_min.setDecimals(decimals)
            self.cutoff_max.setDecimals(decimals)
            self.cutoff_min.setValue(cutoff_min)
            self.cutoff_max.setValue(cutoff_max)

            logger.info(
                f"Grade-Tonnage: Auto-detected cutoff range {cutoff_min:.2f} - {cutoff_max:.2f} "
                f"(grade range: {grade_min:.2f} - {grade_max:.2f}, P5={p5:.2f}, P95={p95:.2f})"
            )

        except Exception as e:
            logger.warning(f"Grade-Tonnage: Could not auto-detect cutoff range: {e}")


# =============================================================================
# ANALYSIS WORKER THREAD
# =============================================================================

class AnalysisWorker(QThread):
    """
    Background worker thread for geostatistical analysis.

    Runs the computationally intensive analysis in a separate thread
    to prevent UI freezing and provide progress updates.
    """

    # Signals
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(object, object)  # grade_tonnage_curve, sensitivity_analysis
    advanced_finished = pyqtSignal(object, object)  # domain_result, classification_result
    error = pyqtSignal(str)  # Error message

    def __init__(self, block_model_data: pd.DataFrame, config: GeostatsGradeTonnageConfig,
                 cutoff_range: np.ndarray, gt_engine: GeostatsGradeTonnageEngine,
                 sensitivity_engine: CutoffSensitivityEngine, column_config: Dict[str, str],
                 advanced_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analysis worker.

        Args:
            block_model_data: Input block model DataFrame
            config: Analysis configuration
            cutoff_range: Array of cutoff grades to analyze
            gt_engine: Grade-tonnage engine instance
            sensitivity_engine: Sensitivity analysis engine instance
            column_config: Dictionary with column names (grade, tonnage, x, y, z)
            advanced_config: Optional advanced analysis configuration
        """
        super().__init__()
        self.block_model_data = block_model_data
        self.config = config
        self.cutoff_range = cutoff_range
        self.gt_engine = gt_engine
        self.sensitivity_engine = sensitivity_engine
        self.column_config = column_config
        self.advanced_config = advanced_config or {}
        
        # Advanced analysis results
        self.domain_result = None
        self.classification_result = None

    def run(self):
        """Execute the analysis in the background thread."""
        try:
            # Step 1: Initialize
            self.progress.emit("🔄 Initializing analysis parameters...")

            # Get column selections from UI configuration
            grade_col = self.column_config['grade']
            tonnage_col = self.column_config['tonnage']
            x_col = self.column_config['x']
            y_col = self.column_config['y']
            z_col = self.column_config['z']
            
            # Apply config to engine
            self.gt_engine.config = self.config
            
            # Determine mode for progress message
            mode_str = "BLOCK MODEL" if self.config.data_mode == DataMode.BLOCK_MODEL else "COMPOSITES"

            # Step 2: Run grade-tonnage analysis
            if self.config.data_mode == DataMode.COMPOSITES:
                self.progress.emit(f"📊 {mode_str} mode: Performing cell-based declustering...")
            else:
                self.progress.emit(f"🧊 {mode_str} mode: Computing grade-tonnage curve (no declustering)...")
            
            grade_tonnage_curve = self.gt_engine.calculate_grade_tonnage_curve(
                self.block_model_data,
                cutoff_range=self.cutoff_range,
                element_column=grade_col,
                tonnage_column=tonnage_col,
                x_column=x_col,
                y_column=y_col,
                z_column=z_col
            )

            # Step 3: Run sensitivity analysis
            self.progress.emit("🎯 Performing cut-off sensitivity analysis...")
            optimization_method = CutoffOptimizationMethod.NPV_MAXIMIZATION  # Could be configurable
            sensitivity_analysis = self.sensitivity_engine.perform_sensitivity_analysis(
                grade_tonnage_curve,
                self.cutoff_range,
                optimization_method
            )

            # Step 4: Advanced Analysis (JORC/SAMREC)
            domain_result = None
            classification_result = None
            
            # Domain Analysis
            if self.advanced_config.get('enable_domain_analysis', False):
                domain_col = self.advanced_config.get('domain_column')
                if domain_col and domain_col in self.block_model_data.columns:
                    self.progress.emit("🏔️ Running domain-wise GT analysis...")
                    try:
                        domain_engine = DomainGTEngine(self.config)
                        domain_result = domain_engine.compute_domain_curves(
                            self.block_model_data,
                            domain_column=domain_col,
                            grade_column=grade_col,
                            tonnage_column=tonnage_col,
                            x_column=x_col,
                            y_column=y_col,
                            z_column=z_col,
                            cutoffs=self.cutoff_range
                        )
                        logger.info(f"Domain analysis complete: {len(domain_result.domains)} domains")
                    except Exception as e:
                        logger.error(f"Domain analysis failed: {e}", exc_info=True)
            
            # Classification Analysis
            if self.advanced_config.get('enable_classification_analysis', False):
                class_col = self.advanced_config.get('classification_column')
                if class_col and class_col in self.block_model_data.columns:
                    self.progress.emit("📊 Running M/I/I classification analysis...")
                    try:
                        class_engine = ClassificationGTEngine(self.config)
                        classification_result = class_engine.compute_classification_curves(
                            self.block_model_data,
                            classification_column=class_col,
                            grade_column=grade_col,
                            tonnage_column=tonnage_col,
                            x_column=x_col,
                            y_column=y_col,
                            z_column=z_col,
                            cutoffs=self.cutoff_range
                        )
                        logger.info("Classification analysis complete")
                    except Exception as e:
                        logger.error(f"Classification analysis failed: {e}", exc_info=True)
            
            # Store advanced results
            self.domain_result = domain_result
            self.classification_result = classification_result

            # Step 5: Finalize
            self.progress.emit("✨ Analysis complete - generating visualizations...")

            # Emit results
            self.finished.emit(grade_tonnage_curve, sensitivity_analysis)
            
            # Emit advanced results if any
            if domain_result or classification_result:
                self.advanced_finished.emit(domain_result, classification_result)

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.exception("Analysis worker error")
            self.error.emit(error_msg)


# =============================================================================
# LEGACY COMPATIBILITY METHODS
# =============================================================================

    def set_block_model(self, bm) -> None:
        """
        Compatibility helper for external callers.
        Forwards to the internal block model loader.
        """
        try:
            logger.info(f"Grade-Tonnage: set_block_model called with {type(bm).__name__}")
            self._on_bm_loaded(bm)
        except Exception as e:
            logger.error(f"GradeTonnagePanel.set_block_model failed: {e}", exc_info=True)

    # Legacy method aliases for backward compatibility
    def _run_analysis(self):
        """Legacy method - forwards to new geostats analysis."""
        if self._comparison_mode:
            self._run_comparison_analysis()
        else:
            self._run_geostats_analysis()

    def _export(self):
        """Legacy method - forwards to new export functionality."""
        self._export_results()

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
        self.setStyleSheet(get_analysis_panel_stylesheet())
