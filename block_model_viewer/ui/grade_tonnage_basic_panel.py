"""
Grade-Tonnage Basic Analysis Panel.

Simplified panel for core grade-tonnage curve generation.
This panel provides the essential GT analysis without economic optimization
or JORC/SAMREC compliance features.

Features:
- Data source selection (block model or classified)
- Grade/tonnage/coordinate column selection
- Cutoff range configuration
- Basic GT curve visualization
- Statistics summary

For economic analysis, use CutoffOptimizationPanel.
For JORC/SAMREC compliance, use ResourceClassificationPanel.

Author: GeoX Mining Software
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QCheckBox,
    QSplitter, QTabWidget, QFormLayout, QFrame, QScrollArea,
    QProgressBar, QTextEdit, QSpinBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QFont

# Matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.ticker
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..mine_planning.cutoff.geostats_grade_tonnage import (
    GeostatsGradeTonnageEngine,
    GeostatsGradeTonnageConfig,
    DataMode,
    GradeWeightingMethod,
    validate_grade_tonnage_data
)

from ..mine_planning.cutoff.advanced_visualization import (
    MiningVisualizationCoordinator,
    PlotConfig,
    ColorScheme
)

from .base_panel import BasePanel
from .comparison_utils import ComparisonColors, SourceSelectionWidget
from .modern_styles import get_theme_colors

logger = logging.getLogger(__name__)


class GradeTonnageBasicPanel(BasePanel):
    """
    Basic Grade-Tonnage Analysis Panel.

    Provides core GT curve generation with minimal complexity.
    For advanced features, use the companion panels.
    """

    # Signals
    analysisCompleted = pyqtSignal(object)  # grade_tonnage_curve
    analysisProgress = pyqtSignal(str)
    analysisError = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._block_model_data = None
        self.grade_tonnage_curve = None
        self._registry_initialized = False

        # Storage for multiple block model sources
        self._block_model_sources: Dict[str, Any] = {}
        self._available_sources: list = []
        self._current_source: str = ""
        self._stored_sgsim_results = None

        # Comparison mode support
        self._comparison_mode: bool = False
        self._comparison_results: Dict[str, Any] = {}  # Store GT curves per source

        # Initialize engine
        self.gt_engine = GeostatsGradeTonnageEngine()
        self.viz_coordinator = MiningVisualizationCoordinator()

        # Analysis state
        self.is_analyzing = False

        self._init_registry()

    def _init_registry(self):
        """Initialize connection to data registry."""
        try:
            self.registry = self.get_registry()
            if self.registry:
                self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
                self.registry.blockModelGenerated.connect(self._on_block_model_loaded)

                # Classified block model
                if hasattr(self.registry, 'blockModelClassified'):
                    self.registry.blockModelClassified.connect(self._on_block_model_classified)

                # SGSIM results
                if hasattr(self.registry, 'sgsimResultsLoaded'):
                    self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)

                # Load all existing sources
                self._load_existing_sources()

                self._registry_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize registry: {e}")

    def _load_existing_sources(self):
        """Load all existing block model sources from registry."""
        if not self.registry:
            return

        # Regular block model
        bm = self.registry.get_block_model()
        if bm is not None:
            self._register_source("Block Model", bm, auto_select=True)

        # Classified block model
        cbm = self.registry.get_classified_block_model()
        if cbm is not None:
            self._register_source("Classified Block Model", cbm, auto_select=not self._available_sources)

        # SGSIM results
        if hasattr(self.registry, 'get_sgsim_results'):
            sgsim = self.registry.get_sgsim_results()
            if sgsim is not None:
                self._register_sgsim_sources(sgsim)

        self._update_source_selector()

    def setup_ui(self):
        """Set up the simplified UI."""
        layout = self.main_layout if hasattr(self, 'main_layout') else QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Configuration
        left_panel = self._create_config_panel()
        splitter.addWidget(left_panel)

        # Right: Results
        right_panel = self._create_results_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)

        layout.addWidget(splitter)

    def _create_config_panel(self) -> QWidget:
        """Create configuration panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setSpacing(10)

        # Data Configuration
        self._create_data_config(config_layout)

        # Cutoff Range
        self._create_cutoff_config(config_layout)

        # Analysis Options
        self._create_analysis_options(config_layout)

        config_layout.addStretch()
        scroll.setWidget(config_widget)
        layout.addWidget(scroll)

        return panel

    def _create_data_config(self, layout):
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
            }
        """)

        form = QFormLayout(group)
        form.setSpacing(8)

        # Block model source selector
        self.source_combo = QComboBox()
        self.source_combo.setToolTip("Select block model source for analysis")
        self.source_combo.addItem("No block model loaded", "none")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Block Model Source:", self.source_combo)

        # Comparison mode widget
        self._source_selection_widget = SourceSelectionWidget()
        self._source_selection_widget.comparison_mode_changed.connect(self._on_comparison_mode_changed)
        self._source_selection_widget.sources_changed.connect(self._on_comparison_sources_changed)
        form.addRow("", self._source_selection_widget)

        # Source info label
        self.source_info_label = QLabel("No block model selected")
        self.source_info_label.setStyleSheet("color: #888; font-style: italic; font-size: 9pt;")
        form.addRow("", self.source_info_label)

        # Grade column
        self.grade_col = QComboBox()
        self.grade_col.setToolTip("Select the grade column for analysis")
        form.addRow("Grade Column:", self.grade_col)

        # Tonnage column with calculation option
        tonnage_layout = QHBoxLayout()
        self.tonnage_col = QComboBox()
        self.tonnage_col.setToolTip("Select the tonnage column, or enable calculation below")
        tonnage_layout.addWidget(self.tonnage_col)

        self.calculate_tonnage_check = QCheckBox("Calculate")
        self.calculate_tonnage_check.setToolTip("Calculate tonnage from block volume and density if tonnage column doesn't exist")
        self.calculate_tonnage_check.toggled.connect(self._on_calculate_tonnage_toggled)
        tonnage_layout.addWidget(self.calculate_tonnage_check)

        form.addRow("Tonnage Column:", tonnage_layout)

        # Tonnage calculation parameters (initially hidden)
        self.tonnage_calc_group = QGroupBox("Tonnage Calculation")
        self.tonnage_calc_group.setVisible(False)
        tonnage_calc_form = QFormLayout(self.tonnage_calc_group)
        tonnage_calc_form.setContentsMargins(10, 5, 10, 5)

        # Density column selector
        self.density_col = QComboBox()
        self.density_col.setToolTip("Select the density/SG column (tonnes/m³)")
        tonnage_calc_form.addRow("Density Column:", self.density_col)

        # Block dimensions
        dim_layout = QHBoxLayout()
        self.block_x_size = QDoubleSpinBox()
        self.block_x_size.setRange(0.01, 10000)
        self.block_x_size.setValue(10.0)
        self.block_x_size.setDecimals(2)
        self.block_x_size.setSuffix(" m")
        dim_layout.addWidget(QLabel("X:"))
        dim_layout.addWidget(self.block_x_size)

        self.block_y_size = QDoubleSpinBox()
        self.block_y_size.setRange(0.01, 10000)
        self.block_y_size.setValue(10.0)
        self.block_y_size.setDecimals(2)
        self.block_y_size.setSuffix(" m")
        dim_layout.addWidget(QLabel("Y:"))
        dim_layout.addWidget(self.block_y_size)

        self.block_z_size = QDoubleSpinBox()
        self.block_z_size.setRange(0.01, 10000)
        self.block_z_size.setValue(10.0)
        self.block_z_size.setDecimals(2)
        self.block_z_size.setSuffix(" m")
        dim_layout.addWidget(QLabel("Z:"))
        dim_layout.addWidget(self.block_z_size)

        tonnage_calc_form.addRow("Block Dimensions:", dim_layout)

        # Info label
        self.tonnage_calc_info = QLabel("Tonnage = (X × Y × Z) × Density")
        self.tonnage_calc_info.setStyleSheet("color: #666; font-style: italic; font-size: 9pt;")
        tonnage_calc_form.addRow("Formula:", self.tonnage_calc_info)

        form.addRow("", self.tonnage_calc_group)

        # Coordinate columns
        self.x_col = QComboBox()
        self.y_col = QComboBox()
        self.z_col = QComboBox()
        form.addRow("X Coordinate:", self.x_col)
        form.addRow("Y Coordinate:", self.y_col)
        form.addRow("Z Coordinate:", self.z_col)

        # Data mode
        mode_layout = QHBoxLayout()
        self.data_mode = QComboBox()
        self.data_mode.addItems(["Block Model", "Composites"])
        self.data_mode.setToolTip(
            "Block Model: No declustering (regular grid)\n"
            "Composites: Cell declustering applied"
        )
        mode_layout.addWidget(self.data_mode)

        self.grade_weighting = QComboBox()
        self.grade_weighting.addItems(["Tonnage-Weighted", "Equal-Weighted"])
        self.grade_weighting.setToolTip("Method for averaging grades")
        mode_layout.addWidget(QLabel("Weighting:"))
        mode_layout.addWidget(self.grade_weighting)

        form.addRow("Data Type:", mode_layout)

        layout.addWidget(group)

    def _create_cutoff_config(self, layout):
        """Create cutoff range configuration."""
        group = QGroupBox("2. Cutoff Range")
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
            }
        """)

        form = QFormLayout(group)

        cutoff_layout = QHBoxLayout()

        self.cutoff_min = QDoubleSpinBox()
        self.cutoff_min.setRange(0, 100000)
        self.cutoff_min.setValue(0)
        self.cutoff_min.setDecimals(3)
        cutoff_layout.addWidget(QLabel("Min:"))
        cutoff_layout.addWidget(self.cutoff_min)

        self.cutoff_max = QDoubleSpinBox()
        self.cutoff_max.setRange(0, 100000)
        self.cutoff_max.setValue(0)
        self.cutoff_max.setDecimals(3)
        cutoff_layout.addWidget(QLabel("Max:"))
        cutoff_layout.addWidget(self.cutoff_max)

        self.cutoff_steps = QSpinBox()
        self.cutoff_steps.setRange(10, 500)
        self.cutoff_steps.setValue(50)
        cutoff_layout.addWidget(QLabel("Steps:"))
        cutoff_layout.addWidget(self.cutoff_steps)

        self.auto_suggest_btn = QPushButton("Auto")
        self.auto_suggest_btn.setToolTip("Auto-suggest cutoff range based on data percentiles")
        self.auto_suggest_btn.setStyleSheet("background-color: #388e3c; color: white;")
        self.auto_suggest_btn.clicked.connect(self._auto_detect_cutoff_range)
        cutoff_layout.addWidget(self.auto_suggest_btn)

        form.addRow("Range:", cutoff_layout)

        # Show uncertainty bands
        self.show_uncertainty = QCheckBox("Show CV Uncertainty Bands")
        self.show_uncertainty.setChecked(True)
        self.show_uncertainty.setToolTip(
            "Display heuristic CV-based uncertainty bands.\n"
            "NOTE: These are NOT formal statistical confidence intervals."
        )
        form.addRow("Display:", self.show_uncertainty)

        layout.addWidget(group)

    def _create_analysis_options(self, layout):
        """Create analysis control buttons."""
        group = QGroupBox("3. Analysis")
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
            }
        """)

        form = QFormLayout(group)

        # Run button
        self.run_btn = QPushButton("Run GT Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4caf50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #ccc; }
        """)
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        form.addRow("", self.run_btn)

        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        form.addRow("", self.export_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        form.addRow("Progress:", self.progress_bar)

        layout.addWidget(group)

    def _create_results_panel(self) -> QWidget:
        """Create results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Tabs for curve and statistics
        self.results_tabs = QTabWidget()

        # Curve tab - with matplotlib chart
        curve_tab = QWidget()
        curve_layout = QVBoxLayout(curve_tab)

        # Create matplotlib figure for GT curve - professional dual-axis design
        if MATPLOTLIB_AVAILABLE:
            colors = get_theme_colors()
            self.figure = Figure(figsize=(10, 6), dpi=100)
            self.figure.patch.set_facecolor(colors.PANEL_BG)
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # Single plot with dual y-axes (industry standard GT curve)
            self.ax_tonnage = self.figure.add_subplot(111)
            self.ax_grade = self.ax_tonnage.twinx()  # Share x-axis

            self._style_axes()

            # Toolbar for interactive navigation (zoom, pan, save)
            toolbar_container = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)

            self.chart_toolbar = NavigationToolbar(self.canvas, self)
            self.chart_toolbar.setStyleSheet(f"""
                QToolBar {{ background: {colors.PANEL_BG}; border: none; spacing: 5px; }}
                QToolButton {{ background: {colors.ELEVATED_BG}; border: 1px solid {colors.BORDER}; border-radius: 4px; padding: 4px; color: {colors.TEXT_PRIMARY}; }}
                QToolButton:hover {{ background: {colors.CARD_HOVER}; }}
                QToolButton:pressed {{ background: {colors.ACCENT_PRIMARY}; }}
            """)
            toolbar_layout.addWidget(self.chart_toolbar)

            # Export chart button
            self.export_chart_btn = QPushButton("Export Chart")
            self.export_chart_btn.setStyleSheet("""
                QPushButton { background: #238636; color: white; border: none; border-radius: 4px; padding: 6px 12px; font-weight: 500; }
                QPushButton:hover { background: #2ea043; }
            """)
            self.export_chart_btn.clicked.connect(self._export_chart)
            toolbar_layout.addWidget(self.export_chart_btn)
            toolbar_layout.addStretch()

            curve_layout.addWidget(toolbar_container)
            curve_layout.addWidget(self.canvas, stretch=3)
        else:
            self.curve_placeholder = QLabel("Matplotlib not available - install matplotlib for charts")
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
            curve_layout.addWidget(self.curve_placeholder)

        # Curve data table (below chart)
        self.curve_table = QTableWidget(0, 6)
        self.curve_table.setHorizontalHeaderLabels([
            "Cutoff", "Tonnage", "Avg Grade", "Metal Qty",
            "CV Lower", "CV Upper"
        ])
        self.curve_table.setMaximumHeight(150)
        curve_layout.addWidget(self.curve_table, stretch=1)

        self.results_tabs.addTab(curve_tab, "GT Curve")

        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setPlainText("Run analysis to display statistics...")
        stats_layout.addWidget(self.stats_text)

        self.results_tabs.addTab(stats_tab, "Statistics")

        layout.addWidget(self.results_tabs)

        # Status
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(60)
        self.status_text.setReadOnly(True)
        self.status_text.setPlainText("Ready for analysis...")
        layout.addWidget(self.status_text)

        return panel

    def on_block_model_changed(self):
        """Called when block model is set via set_block_model()."""
        if self._block_model is not None:
            self._register_source("Block Model (Set)", self._block_model, auto_select=True)
            self._update_source_selector()

    def _on_block_model_loaded(self, block_model):
        """Handle block model loaded from registry signal."""
        self._register_source("Block Model", block_model, auto_select=not self._available_sources)
        self._update_source_selector()

    def _on_block_model_classified(self, block_model):
        """Handle classified block model from registry."""
        self._register_source("Classified Block Model", block_model, auto_select=True)
        self._update_source_selector()

    def _on_sgsim_loaded(self, results):
        """Handle SGSIM results - register individual statistics as separate sources."""
        if results is None:
            return
        self._stored_sgsim_results = results
        self._register_sgsim_sources(results)
        self._update_source_selector()

    def _register_source(self, name: str, data, auto_select: bool = False):
        """Register a block model source."""
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

        logger.info(f"GT Basic: Registered source '{name}' ({len(df):,} blocks)")

        if auto_select and (not self._current_source or self._current_source == ""):
            self._current_source = name
            self._on_bm_loaded_internal(df)

    def _register_sgsim_sources(self, sgsim_results):
        """Register SGSIM results as multiple block model sources.

        SGSIM stores individual statistics in results['summary'] dict:
        - mean, std, p10, p50, p90 as numpy arrays
        Grid cell_data typically only has the E-type mean property.
        """
        import pyvista as pv
        import numpy as np

        if sgsim_results is None:
            return

        if not isinstance(sgsim_results, dict):
            logger.warning(f"GT Basic: SGSIM results is not a dict, type={type(sgsim_results)}")
            return

        variable = sgsim_results.get('variable', 'Grade')
        summary = sgsim_results.get('summary', {})
        params = sgsim_results.get('params')
        grid = sgsim_results.get('grid') or sgsim_results.get('pyvista_grid')

        logger.info(f"GT Basic: SGSIM results keys: {list(sgsim_results.keys())}")
        logger.info(f"GT Basic: Summary keys: {list(summary.keys()) if summary else 'None'}")
        logger.info(f"GT Basic: params = {params is not None}")

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
                    logger.info(f"GT Basic: Extracted {n_blocks:,} cell centers from grid")

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
                logger.info(f"GT Basic: Generated {n_blocks:,} cell centers from params ({nx}x{ny}x{nz})")
            except Exception as e:
                logger.warning(f"GT Basic: Failed to generate coords from params: {e}")

        if base_df is None or base_df.empty:
            logger.warning("GT Basic: Could not extract coordinates from SGSIM grid or params")
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
                    logger.info(f"GT Basic: Registered {display_prefix} ({variable})")

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
            logger.info(f"GT Basic: Registered {len(found_stats)} SGSIM statistics: {found_stats}")

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

        # Update comparison widget
        if hasattr(self, '_source_selection_widget'):
            self._update_comparison_sources()

    def _on_source_changed(self, index: int):
        """Handle block model source selection change."""
        if index < 0 or not hasattr(self, 'source_combo'):
            return

        source_name = self.source_combo.itemData(index)
        if source_name is None or source_name == "none":
            self._block_model_data = None
            self._current_source = ""
            self.source_info_label.setText("No block model selected")
            return

        if source_name in self._block_model_sources:
            self._current_source = source_name
            df = self._block_model_sources[source_name]

            cols = [c for c in df.columns if c.upper() not in ('X', 'Y', 'Z')]
            self.source_info_label.setText(f"Properties: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")

            self._on_bm_loaded_internal(df)
            logger.info(f"GT Basic: Switched to '{source_name}' ({len(df):,} blocks)")

    def _on_bm_loaded_internal(self, bm):
        """Internal method to process loaded block model (bypasses registration)."""
        if bm is None:
            return

        try:
            if hasattr(bm, 'to_dataframe'):
                self._block_model_data = bm.to_dataframe()
            elif isinstance(bm, pd.DataFrame):
                self._block_model_data = bm
            else:
                return

            self._update_column_selectors()

        except Exception as e:
            logger.error(f"Error loading block model: {e}")

    def _on_bm_loaded(self, bm):
        """Process loaded block model - delegates to internal method."""
        self._on_bm_loaded_internal(bm)

    def _update_column_selectors(self):
        """Update column selectors with available columns."""
        if self._block_model_data is None:
            return

        numeric_cols = self._block_model_data.select_dtypes(
            include=[float, 'int64', 'int32', 'int']
        ).columns.tolist()

        for selector in [self.grade_col, self.tonnage_col,
                        self.x_col, self.y_col, self.z_col, self.density_col]:
            current = selector.currentText()
            selector.clear()
            selector.addItems(numeric_cols)
            if current in numeric_cols:
                selector.setCurrentText(current)

        # Auto-select likely columns
        self._auto_select_columns(numeric_cols)

        # Auto-detect block dimensions from metadata if available
        self._auto_detect_block_dimensions()

        self.run_btn.setEnabled(True)

    def _auto_select_columns(self, cols):
        """Auto-select likely column names."""
        patterns = {
            'grade': ['GRADE', 'AU', 'AG', 'CU', 'FE', 'ZN', '_EST', '_OK'],
            'tonnage': ['TONNAGE', 'TONNES', 'TONS', 'WEIGHT'],
            'density': ['DENSITY', 'SG', 'SPECIFIC_GRAVITY', 'DENS', 'RHO'],
            'x': ['XC', 'X', 'XCENTRE', 'EAST', 'EASTING'],
            'y': ['YC', 'Y', 'YCENTRE', 'NORTH', 'NORTHING'],
            'z': ['ZC', 'Z', 'ZCENTRE', 'RL', 'ELEV', 'ELEVATION']
        }

        selectors = {
            'grade': self.grade_col,
            'tonnage': self.tonnage_col,
            'density': self.density_col,
            'x': self.x_col,
            'y': self.y_col,
            'z': self.z_col
        }

        for key, selector in selectors.items():
            for col in cols:
                if any(p in col.upper() for p in patterns[key]):
                    selector.setCurrentText(col)
                    break

        # Auto-enable tonnage calculation if no tonnage column found
        tonnage_found = any(any(p in col.upper() for p in patterns['tonnage']) for col in cols)
        density_found = any(any(p in col.upper() for p in patterns['density']) for col in cols)

        if not tonnage_found and density_found:
            self.calculate_tonnage_check.setChecked(True)
            self.status_text.setPlainText(
                "No tonnage column found. Auto-enabled tonnage calculation from density.\n"
                "Verify block dimensions and density column before running analysis."
            )

        # Auto-detect cutoff range based on grade data
        self._auto_detect_cutoff_range()

        # Connect grade column change to update cutoff range (disconnect first to avoid duplicates)
        try:
            self.grade_col.currentTextChanged.disconnect(self._auto_detect_cutoff_range)
        except TypeError:
            # Signal wasn't connected yet, this is fine
            pass
        self.grade_col.currentTextChanged.connect(self._auto_detect_cutoff_range)

    def _on_calculate_tonnage_toggled(self, checked: bool):
        """Handle tonnage calculation checkbox toggle."""
        self.tonnage_calc_group.setVisible(checked)

        if checked:
            # Disable tonnage column selector (will be calculated)
            self.tonnage_col.setEnabled(False)
            self.status_text.setPlainText(
                "Tonnage will be calculated from block volume × density.\n"
                "Verify block dimensions and density column below."
            )
        else:
            # Re-enable tonnage column selector
            self.tonnage_col.setEnabled(True)
            self.status_text.setPlainText("Select tonnage column from block model.")

    def _auto_detect_block_dimensions(self):
        """Auto-detect block dimensions from block model metadata if available."""
        if self._block_model_data is None:
            return

        # Try to detect from current source metadata
        current_source = self._block_model_sources.get(self._current_source)
        if current_source is None:
            return

        # Check for common metadata attributes
        try:
            # For BlockModel objects
            if hasattr(current_source, 'metadata'):
                metadata = current_source.metadata
                if hasattr(metadata, 'xinc'):
                    self.block_x_size.setValue(metadata.xinc)
                    self.block_y_size.setValue(metadata.yinc)
                    self.block_z_size.setValue(metadata.zinc)
                    logger.info(f"GT Basic: Auto-detected block dimensions: {metadata.xinc} × {metadata.yinc} × {metadata.zinc}")
                    return

            # Try to infer from coordinate spacing
            if 'X' in current_source.columns and 'Y' in current_source.columns and 'Z' in current_source.columns:
                x_unique = np.sort(current_source['X'].unique())
                y_unique = np.sort(current_source['Y'].unique())
                z_unique = np.sort(current_source['Z'].unique())

                # Initialize default values
                x_mode = y_mode = z_mode = None

                if len(x_unique) > 1:
                    x_spacing = np.diff(x_unique)
                    x_mode = np.median(x_spacing[x_spacing > 0])
                    if x_mode > 0:
                        self.block_x_size.setValue(x_mode)

                if len(y_unique) > 1:
                    y_spacing = np.diff(y_unique)
                    y_mode = np.median(y_spacing[y_spacing > 0])
                    if y_mode > 0:
                        self.block_y_size.setValue(y_mode)

                if len(z_unique) > 1:
                    z_spacing = np.diff(z_unique)
                    z_mode = np.median(z_spacing[z_spacing > 0])
                    if z_mode > 0:
                        self.block_z_size.setValue(z_mode)

                # Only log if we successfully inferred at least one dimension
                if x_mode or y_mode or z_mode:
                    dims = f"{x_mode:.2f} × {y_mode:.2f} × {z_mode:.2f}" if all([x_mode, y_mode, z_mode]) else "partial detection"
                    logger.info(f"GT Basic: Inferred block dimensions from spacing: {dims}")

        except Exception as e:
            logger.debug(f"Could not auto-detect block dimensions: {e}")

    def _calculate_tonnage_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tonnage column from block volume and density.

        Args:
            df: Block model DataFrame

        Returns:
            DataFrame with added 'TONNAGE_CALC' column
        """
        density_col = self.density_col.currentText()

        if not density_col or density_col not in df.columns:
            raise ValueError(f"Density column '{density_col}' not found in block model.")

        # Check for NaN values in density
        nan_count = df[density_col].isna().sum()
        if nan_count > 0:
            logger.warning(f"GT Basic: {nan_count} blocks have NaN density values - these will be excluded from analysis")

        # Check for invalid density values (zero or negative)
        invalid_density = df[~df[density_col].isna() & (df[density_col] <= 0)]
        if len(invalid_density) > 0:
            raise ValueError(
                f"Density column '{density_col}' contains {len(invalid_density)} blocks with zero or negative values.\n"
                f"Density must be positive (typical range: 1.5 - 5.0 tonnes/m³).\n"
                f"Invalid density range: {df[density_col].min():.3f} - {df[density_col].max():.3f}"
            )

        # Get block dimensions
        block_x = self.block_x_size.value()
        block_y = self.block_y_size.value()
        block_z = self.block_z_size.value()

        # Validate block dimensions
        if block_x <= 0 or block_y <= 0 or block_z <= 0:
            raise ValueError(
                f"Block dimensions must be positive.\n"
                f"Current values: X={block_x}, Y={block_y}, Z={block_z}"
            )

        # Calculate volume (m³)
        volume = block_x * block_y * block_z

        # Calculate tonnage = volume × density
        # Note: Assumes density is in tonnes/m³ (same as SG for most minerals)
        df['TONNAGE_CALC'] = volume * df[density_col]

        # Log statistics
        valid_tonnage = df['TONNAGE_CALC'].dropna()
        if len(valid_tonnage) > 0:
            logger.info(
                f"GT Basic: Calculated tonnage from volume ({block_x}×{block_y}×{block_z} = {volume:.2f} m³) "
                f"× density column '{density_col}'"
            )
            logger.info(
                f"GT Basic: Tonnage range: {valid_tonnage.min():.2f} - {valid_tonnage.max():.2f} tonnes "
                f"(mean: {valid_tonnage.mean():.2f}, {len(valid_tonnage):,} valid blocks)"
            )
        else:
            raise ValueError("No valid tonnage values calculated - check density column for issues")

        return df

    def _auto_detect_cutoff_range(self):
        """Auto-detect appropriate cutoff range based on grade column statistics."""
        if self._block_model_data is None:
            return

        grade_col = self.grade_col.currentText()
        if not grade_col or grade_col not in self._block_model_data.columns:
            return

        try:
            grades = self._block_model_data[grade_col].dropna()
            if len(grades) == 0:
                return

            # Use percentiles for robust range detection
            p5 = grades.quantile(0.05)
            p95 = grades.quantile(0.95)
            grade_min = grades.min()
            grade_max = grades.max()
            grade_mean = grades.mean()

            # Determine sensible cutoff range
            # Start from near minimum or 0, end at around P90-P95
            if grade_min >= 0:
                # For grades like Fe (20-60%), Au (0.5-10 g/t), Cu (0.2-2%)
                # Start cutoff at ~P5 or a bit below minimum
                cutoff_min = max(0, p5 * 0.8)  # Start at 80% of P5 or 0
                cutoff_max = p95  # End at P95

                # Round to sensible values
                if grade_max > 10:
                    # Higher grade values (like Fe %): round to nearest integer
                    cutoff_min = np.floor(cutoff_min)
                    cutoff_max = np.ceil(cutoff_max)
                elif grade_max > 1:
                    # Medium grade values: round to 1 decimal
                    cutoff_min = np.floor(cutoff_min * 10) / 10
                    cutoff_max = np.ceil(cutoff_max * 10) / 10
                else:
                    # Low grade values (like Au g/t): round to 2 decimals
                    cutoff_min = np.floor(cutoff_min * 100) / 100
                    cutoff_max = np.ceil(cutoff_max * 100) / 100

                # Update spinboxes
                self.cutoff_min.setValue(cutoff_min)
                self.cutoff_max.setValue(cutoff_max)

                # Update status with detected range info
                self.status_text.setPlainText(
                    f"Auto-detected cutoff range for '{grade_col}':\n"
                    f"Grade range: {grade_min:.2f} - {grade_max:.2f} (mean: {grade_mean:.2f})\n"
                    f"Recommended cutoffs: {cutoff_min:.2f} - {cutoff_max:.2f}"
                )

        except Exception as e:
            logger.warning(f"Could not auto-detect cutoff range: {e}")

    def _run_analysis(self):
        """Run grade-tonnage analysis."""
        # Check for comparison mode
        if self._comparison_mode:
            self._run_comparison_analysis()
            return

        if self._block_model_data is None:
            QMessageBox.warning(self, "No Data", "Load a block model first.")
            return

        try:
            self.run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.status_text.setPlainText("Starting analysis...")

            # Prepare data - calculate tonnage if needed
            analysis_data = self._block_model_data.copy()

            # Determine tonnage column
            tonnage_column = self.tonnage_col.currentText()

            if self.calculate_tonnage_check.isChecked():
                self.status_text.setPlainText("Calculating tonnage from block volume and density...")
                try:
                    analysis_data = self._calculate_tonnage_column(analysis_data)
                    tonnage_column = 'TONNAGE_CALC'
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Tonnage Calculation Error",
                        f"Failed to calculate tonnage:\n{str(e)}\n\nPlease verify density column and block dimensions."
                    )
                    return

            self.progress_bar.setValue(20)

            # Collect configuration
            config = GeostatsGradeTonnageConfig(
                data_mode=DataMode.COMPOSITES if self.data_mode.currentIndex() == 1 else DataMode.BLOCK_MODEL,
                grade_weighting=GradeWeightingMethod.TONNAGE_WEIGHTED if self.grade_weighting.currentIndex() == 0 else GradeWeightingMethod.EQUAL_WEIGHT
            )

            self.gt_engine.config = config

            cutoff_range = np.linspace(
                self.cutoff_min.value(),
                self.cutoff_max.value(),
                self.cutoff_steps.value()
            )

            self.progress_bar.setValue(30)

            # Run analysis
            self.grade_tonnage_curve = self.gt_engine.calculate_grade_tonnage_curve(
                analysis_data,
                cutoff_range=cutoff_range,
                element_column=self.grade_col.currentText(),
                tonnage_column=tonnage_column,
                x_column=self.x_col.currentText(),
                y_column=self.y_col.currentText(),
                z_column=self.z_col.currentText()
            )

            self.progress_bar.setValue(80)

            # Update display
            self._update_results_display()

            self.progress_bar.setValue(100)
            self.status_text.setPlainText("Analysis complete!")
            self.export_btn.setEnabled(True)

            self.analysisCompleted.emit(self.grade_tonnage_curve)

        except Exception as e:
            self.status_text.setPlainText(f"Error: {str(e)}")
            logger.exception("Analysis error")

        finally:
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _style_axes(self):
        """Apply theme-aware styling to matplotlib axes."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Get current theme colors
        colors = get_theme_colors()
        bg_color = colors.CARD_BG
        text_color = colors.TEXT_PRIMARY
        grid_color = colors.BORDER
        tonnage_color = '#58a6ff'  # Blue for tonnage (semantic)
        grade_color = '#f0883e'    # Gold/orange for grade (semantic)

        # Style primary axis (tonnage)
        self.ax_tonnage.set_facecolor(bg_color)
        self.ax_tonnage.tick_params(axis='y', colors=tonnage_color, labelsize=10)
        self.ax_tonnage.tick_params(axis='x', colors=text_color, labelsize=10)
        self.ax_tonnage.yaxis.label.set_color(tonnage_color)
        self.ax_tonnage.xaxis.label.set_color(text_color)
        for spine in ['top', 'right']:
            self.ax_tonnage.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.ax_tonnage.spines[spine].set_color(grid_color)

        # Style secondary axis (grade)
        self.ax_grade.tick_params(axis='y', colors=grade_color, labelsize=10)
        self.ax_grade.yaxis.label.set_color(grade_color)
        self.ax_grade.spines['right'].set_color(grade_color)
        self.ax_grade.spines['right'].set_linewidth(1.5)
        for spine in ['top', 'left', 'bottom']:
            self.ax_grade.spines[spine].set_visible(False)

        self.figure.tight_layout(pad=2.0)

    def _update_results_display(self):
        """Update results display with analysis results."""
        if self.grade_tonnage_curve is None:
            return

        points = self.grade_tonnage_curve.points

        # Update table
        self.curve_table.setRowCount(len(points))

        for i, pt in enumerate(points):
            self.curve_table.setItem(i, 0, QTableWidgetItem(f"{pt.cutoff_grade:.2f}"))
            self.curve_table.setItem(i, 1, QTableWidgetItem(f"{pt.tonnage:,.0f}"))
            self.curve_table.setItem(i, 2, QTableWidgetItem(f"{pt.avg_grade:.3f}"))
            self.curve_table.setItem(i, 3, QTableWidgetItem(f"{pt.metal_quantity:,.0f}"))
            self.curve_table.setItem(i, 4, QTableWidgetItem(f"{pt.cv_uncertainty_band[0]:,.0f}"))
            self.curve_table.setItem(i, 5, QTableWidgetItem(f"{pt.cv_uncertainty_band[1]:,.0f}"))

        self.curve_table.setVisible(True)

        # Plot the GT curve
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'figure'):
            self._plot_gt_curve(points)

        # Update statistics
        stats = self.grade_tonnage_curve.global_statistics
        grade_stats = stats.get('grade_statistics', {})

        stats_text = f"""
GRADE-TONNAGE ANALYSIS SUMMARY
{'='*40}

Total Tonnage: {stats.get('total_tonnage', 0):,.0f} tonnes
Total Metal: {stats.get('total_metal', 0):,.0f} units
Sample Count: {stats.get('sample_count', 0):,}

GRADE STATISTICS
{'='*40}
Mean Grade: {grade_stats.get('mean', 0):.3f}
Median Grade: {grade_stats.get('median', 0):.3f}
Std Dev: {grade_stats.get('std', 0):.3f}
CV: {grade_stats.get('cv', 0):.2f}
Min: {grade_stats.get('min', 0):.3f}
Max: {grade_stats.get('max', 0):.3f}

Data Mode: {stats.get('data_mode', 'block_model')}
Weighting: {grade_stats.get('weighting_method', 'tonnage-weighted')}
"""
        self.stats_text.setPlainText(stats_text)

    def _plot_gt_curve(self, points):
        """Plot professional dual-axis grade-tonnage curve."""
        if not points:
            return

        # Extract data
        cutoffs = np.array([pt.cutoff_grade for pt in points])
        tonnages = np.array([pt.tonnage for pt in points])
        grades = np.array([pt.avg_grade for pt in points])
        cv_lower = np.array([pt.cv_uncertainty_band[0] for pt in points])
        cv_upper = np.array([pt.cv_uncertainty_band[1] for pt in points])

        # Professional color scheme
        bg_color = '#0d1117'
        text_color = '#c9d1d9'
        grid_color = '#21262d'
        tonnage_color = '#58a6ff'
        tonnage_fill = '#1f6feb'
        grade_color = '#f0883e'
        uncertainty_color = '#388bfd'

        # Clear previous plots
        self.ax_tonnage.clear()
        self.ax_grade.clear()

        # Plot uncertainty band (subtle gradient effect)
        if self.show_uncertainty.isChecked():
            self.ax_tonnage.fill_between(
                cutoffs, cv_lower, cv_upper,
                alpha=0.15, color=uncertainty_color,
                linewidth=0
            )
            # Add subtle edge lines
            self.ax_tonnage.plot(cutoffs, cv_lower, '--', color=uncertainty_color,
                                alpha=0.4, linewidth=0.8)
            self.ax_tonnage.plot(cutoffs, cv_upper, '--', color=uncertainty_color,
                                alpha=0.4, linewidth=0.8)

        # Plot tonnage curve with area fill
        self.ax_tonnage.fill_between(cutoffs, 0, tonnages, alpha=0.2, color=tonnage_fill)
        line_tonnage, = self.ax_tonnage.plot(
            cutoffs, tonnages, '-', color=tonnage_color,
            linewidth=2.5, label='Tonnage', solid_capstyle='round'
        )

        # Plot grade curve
        line_grade, = self.ax_grade.plot(
            cutoffs, grades, '-', color=grade_color,
            linewidth=2.5, label='Average Grade', solid_capstyle='round'
        )

        # Add markers at key points (not every point - cleaner look)
        n_markers = min(10, len(cutoffs))
        marker_indices = np.linspace(0, len(cutoffs)-1, n_markers, dtype=int)
        self.ax_tonnage.scatter(
            cutoffs[marker_indices], tonnages[marker_indices],
            color=tonnage_color, s=40, zorder=5, edgecolors='white', linewidths=1
        )
        self.ax_grade.scatter(
            cutoffs[marker_indices], grades[marker_indices],
            color=grade_color, s=40, zorder=5, edgecolors='white', linewidths=1
        )

        # Styling
        self.ax_tonnage.set_facecolor(bg_color)
        self.ax_tonnage.set_xlabel('Cutoff Grade', fontsize=11, color=text_color, fontweight='medium')
        self.ax_tonnage.set_ylabel('Tonnage', fontsize=11, color=tonnage_color, fontweight='medium')
        self.ax_grade.set_ylabel('Average Grade', fontsize=11, color=grade_color, fontweight='medium')
        # Ensure the grade axis label appears on the right side
        self.ax_grade.yaxis.set_label_position('right')

        # Title
        self.ax_tonnage.set_title(
            'Grade-Tonnage Curve', fontsize=13, color=text_color,
            fontweight='bold', pad=15
        )

        # Grid (subtle)
        self.ax_tonnage.grid(True, alpha=0.15, color=grid_color, linestyle='-', linewidth=0.5)
        self.ax_tonnage.set_axisbelow(True)

        # Format tonnage axis (auto-scale to M/B)
        def format_tonnage(x, p):
            if x >= 1e9:
                return f'{x/1e9:.1f}B'
            elif x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.0f}K'
            return f'{x:.0f}'

        self.ax_tonnage.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(format_tonnage)
        )

        # Tick styling
        self.ax_tonnage.tick_params(axis='y', colors=tonnage_color, labelsize=10)
        self.ax_tonnage.tick_params(axis='x', colors=text_color, labelsize=10)
        self.ax_grade.tick_params(axis='y', colors=grade_color, labelsize=10)

        # Spine styling
        for spine in ['top']:
            self.ax_tonnage.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.ax_tonnage.spines[spine].set_color(grid_color)
            self.ax_tonnage.spines[spine].set_linewidth(0.8)
        self.ax_tonnage.spines['right'].set_visible(False)

        self.ax_grade.spines['right'].set_color(grade_color)
        self.ax_grade.spines['right'].set_linewidth(1.5)
        for spine in ['top', 'left', 'bottom']:
            self.ax_grade.spines[spine].set_visible(False)

        # Combined legend
        lines = [line_tonnage, line_grade]
        labels = ['Tonnage (left axis)', 'Avg Grade (right axis)']
        legend = self.ax_tonnage.legend(
            lines, labels, loc='upper right', frameon=True,
            facecolor='#161b22', edgecolor=grid_color,
            fontsize=9, labelcolor=text_color
        )
        legend.get_frame().set_alpha(0.9)

        # Set axis limits with padding
        self.ax_tonnage.set_xlim(cutoffs.min(), cutoffs.max())
        self.ax_tonnage.set_ylim(0, tonnages.max() * 1.1)
        self.ax_grade.set_ylim(grades.min() * 0.95, grades.max() * 1.05)

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

    def _export_results(self):
        """Export analysis results to CSV."""
        if self.grade_tonnage_curve is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )

        if filename:
            try:
                data = []
                for pt in self.grade_tonnage_curve.points:
                    data.append({
                        'cutoff_grade': pt.cutoff_grade,
                        'tonnage': pt.tonnage,
                        'avg_grade': pt.avg_grade,
                        'metal_quantity': pt.metal_quantity,
                        'cv_lower': pt.cv_uncertainty_band[0],
                        'cv_upper': pt.cv_uncertainty_band[1]
                    })

                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

                QMessageBox.information(self, "Export Complete", f"Results exported to {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # ------------------------------------------------------------------
    # Comparison mode methods
    # ------------------------------------------------------------------

    def _on_comparison_mode_changed(self, enabled: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = enabled
        # Hide single source selector when in comparison mode
        self.source_combo.setVisible(not enabled)
        self.source_info_label.setVisible(not enabled)
        logger.info(f"GT Basic: Comparison mode {'enabled' if enabled else 'disabled'}")

        if enabled:
            self._update_comparison_sources()

    def _on_comparison_sources_changed(self, selected_keys: List[str]):
        """Handle comparison source selection changes."""
        logger.info(f"GT Basic: Comparison sources changed: {selected_keys}")

        # Populate grade column from selected sources
        if selected_keys:
            self._populate_comparison_properties(selected_keys)

        # Enable run button if at least 2 sources selected
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(len(selected_keys) >= 2)

    def _populate_comparison_properties(self, selected_keys: List[str]):
        """Populate grade column dropdown with properties from selected sources."""
        import pandas as pd

        all_properties = set()

        for source_name in selected_keys:
            df = self._block_model_sources.get(source_name)
            if df is not None:
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]) and col.upper() not in ('X', 'Y', 'Z'):
                        all_properties.add(col)

        # Update grade column dropdown
        if hasattr(self, 'grade_col'):
            current = self.grade_col.currentText()
            self.grade_col.blockSignals(True)
            self.grade_col.clear()

            sorted_props = sorted(all_properties)
            self.grade_col.addItems(sorted_props)

            # Try to restore previous selection or select first grade-like property
            if current and current in sorted_props:
                self.grade_col.setCurrentText(current)
            else:
                for prop in sorted_props:
                    if any(k in prop.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')):
                        self.grade_col.setCurrentText(prop)
                        break

            self.grade_col.blockSignals(False)
            logger.info(f"GT Basic: Populated {len(sorted_props)} properties for comparison")

    def _update_comparison_sources(self):
        """Update the comparison widget with available sources."""
        sources = {}

        for source_name in self._available_sources:
            df = self._block_model_sources.get(source_name)
            if df is not None:
                sources[source_name] = {
                    'display_name': source_name,
                    'block_count': len(df),
                    'df': df
                }

        self._source_selection_widget.update_sources(sources)

    def _run_comparison_analysis(self):
        """Run GT analysis for multiple sources and plot comparison."""
        selected_keys = self._source_selection_widget.get_selected_sources()

        if len(selected_keys) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 sources for comparison.")
            return

        try:
            self.run_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            self.status_text.setPlainText("Starting comparison analysis...")

            # Collect configuration
            config = GeostatsGradeTonnageConfig(
                data_mode=DataMode.COMPOSITES if self.data_mode.currentIndex() == 1 else DataMode.BLOCK_MODEL,
                grade_weighting=GradeWeightingMethod.TONNAGE_WEIGHTED if self.grade_weighting.currentIndex() == 0 else GradeWeightingMethod.EQUAL_WEIGHT
            )

            cutoff_range = np.linspace(
                self.cutoff_min.value(),
                self.cutoff_max.value(),
                self.cutoff_steps.value()
            )

            self._comparison_results = {}
            progress_per_source = 60 / len(selected_keys)

            for i, source_name in enumerate(selected_keys):
                df = self._block_model_sources.get(source_name)
                if df is None:
                    continue

                self.status_text.setPlainText(f"Analyzing: {source_name}...")
                self.progress_bar.setValue(int(10 + i * progress_per_source))

                # Find grade column with flexible matching
                grade_col = self.grade_col.currentText()
                if grade_col not in df.columns:
                    found_match = False

                    # Strategy 1: Column contains grade_col
                    for col in df.columns:
                        if grade_col.upper() in col.upper():
                            grade_col = col
                            found_match = True
                            logger.debug(f"GT Basic: Matched {self.grade_col.currentText()} -> {col} (column contains)")
                            break

                    # Strategy 2: grade_col contains column name
                    if not found_match:
                        for col in df.columns:
                            if col.upper() in grade_col.upper() and col.upper() not in ('X', 'Y', 'Z'):
                                grade_col = col
                                found_match = True
                                logger.debug(f"GT Basic: Matched {self.grade_col.currentText()} -> {col} (grade contains column)")
                                break

                    # Strategy 3: Base name match
                    if not found_match:
                        base_name = grade_col.split('_')[0] if '_' in grade_col else grade_col
                        for col in df.columns:
                            col_base = col.split('_')[0] if '_' in col else col
                            if base_name.upper() == col_base.upper() and col.upper() not in ('X', 'Y', 'Z'):
                                grade_col = col
                                found_match = True
                                logger.debug(f"GT Basic: Matched {self.grade_col.currentText()} -> {col} (base name match)")
                                break

                    # Strategy 4: Any grade-like property
                    if not found_match:
                        for col in df.columns:
                            if any(k in col.upper() for k in ('GRADE', 'FE', 'AU', 'CU', 'ZN', 'AG')) and col.upper() not in ('X', 'Y', 'Z'):
                                grade_col = col
                                found_match = True
                                logger.debug(f"GT Basic: Using grade-like property {col} for {source_name}")
                                break

                    if not found_match:
                        logger.warning(f"GT Basic: No matching grade column for '{self.grade_col.currentText()}' in {source_name}. Available: {list(df.columns)[:10]}")
                        continue

                try:
                    # Prepare data for this source - calculate tonnage if needed
                    analysis_df = df.copy()
                    tonnage_column = self.tonnage_col.currentText()

                    # Calculate tonnage if enabled and density column exists
                    if self.calculate_tonnage_check.isChecked():
                        density_col = self.density_col.currentText()
                        if density_col in analysis_df.columns:
                            try:
                                # Use the validated calculation method for consistency
                                analysis_df = self._calculate_tonnage_column(analysis_df)
                                tonnage_column = 'TONNAGE_CALC'
                                logger.debug(f"GT Basic: Calculated tonnage for {source_name}")
                            except ValueError as e:
                                # Validation error - skip this source with clear message
                                logger.warning(f"GT Basic: Tonnage calculation failed for {source_name}: {e}")
                                continue
                            except Exception as e:
                                logger.warning(f"GT Basic: Failed to calculate tonnage for {source_name}: {e}")
                                continue
                        else:
                            logger.warning(f"GT Basic: Density column '{density_col}' not in {source_name}, skipping tonnage calculation")

                    # Fallback if tonnage column not in dataframe
                    if tonnage_column not in analysis_df.columns:
                        tonnage_column = None

                    # Smart coordinate column detection with fallback
                    def find_coord_column(preferred: str, patterns: list, df_cols: list) -> str:
                        """Find coordinate column with smart fallback."""
                        # Try preferred first
                        if preferred in df_cols:
                            return preferred
                        # Try pattern matching
                        for col in df_cols:
                            if any(p in col.upper() for p in patterns):
                                return col
                        # Last resort: return preferred (will error if missing, which is correct)
                        return preferred

                    x_patterns = ['XC', 'X', 'XCENTRE', 'EAST', 'EASTING']
                    y_patterns = ['YC', 'Y', 'YCENTRE', 'NORTH', 'NORTHING']
                    z_patterns = ['ZC', 'Z', 'ZCENTRE', 'RL', 'ELEV', 'ELEVATION']

                    x_col = find_coord_column(self.x_col.currentText(), x_patterns, list(analysis_df.columns))
                    y_col = find_coord_column(self.y_col.currentText(), y_patterns, list(analysis_df.columns))
                    z_col = find_coord_column(self.z_col.currentText(), z_patterns, list(analysis_df.columns))

                    gt_engine = GeostatsGradeTonnageEngine()
                    gt_engine.config = config

                    curve = gt_engine.calculate_grade_tonnage_curve(
                        analysis_df,
                        cutoff_range=cutoff_range,
                        element_column=grade_col,
                        tonnage_column=tonnage_column,
                        x_column=x_col,
                        y_column=y_col,
                        z_column=z_col
                    )

                    if curve is not None:
                        self._comparison_results[source_name] = {
                            'curve': curve,
                            'grade_col': grade_col
                        }
                        logger.info(f"GT Basic: Computed GT curve for {source_name}")

                except Exception as e:
                    logger.warning(f"GT Basic: Failed to compute GT for {source_name}: {e}")
                    continue

            self.progress_bar.setValue(80)

            if len(self._comparison_results) < 2:
                QMessageBox.warning(self, "Analysis Error", "Could not compute GT curves for at least 2 sources.")
                return

            # Plot comparison
            self._plot_comparison_gt_curves()

            self.progress_bar.setValue(100)
            self.status_text.setPlainText(f"Comparison complete! {len(self._comparison_results)} curves plotted.")
            self.export_btn.setEnabled(True)

        except Exception as e:
            self.status_text.setPlainText(f"Error: {str(e)}")
            logger.exception("Comparison analysis error")

        finally:
            self.run_btn.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _plot_comparison_gt_curves(self):
        """Plot overlaid GT curves for multiple sources."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'figure'):
            return

        # Professional color scheme
        bg_color = '#0d1117'
        text_color = '#c9d1d9'
        grid_color = '#21262d'

        # Clear previous plots
        self.ax_tonnage.clear()
        self.ax_grade.clear()

        # Store lines for legend
        tonnage_lines = []
        grade_lines = []

        for i, (source_name, result) in enumerate(self._comparison_results.items()):
            curve = result['curve']
            points = curve.points

            # Extract data
            cutoffs = np.array([pt.cutoff_grade for pt in points])
            tonnages = np.array([pt.tonnage for pt in points])
            grades = np.array([pt.avg_grade for pt in points])

            # Get color and style from comparison palette
            style = ComparisonColors.get_style(i)
            color = style['color']
            linestyle = style['linestyle']

            # Plot tonnage (solid line)
            line_t, = self.ax_tonnage.plot(
                cutoffs, tonnages,
                color=color, linewidth=2.5, linestyle=linestyle,
                label=f'{source_name} (Tonnage)'
            )
            tonnage_lines.append((line_t, f'{source_name}'))

            # Plot grade (dashed line on secondary axis)
            line_g, = self.ax_grade.plot(
                cutoffs, grades,
                color=color, linewidth=2, linestyle='--',
                alpha=0.8
            )
            grade_lines.append((line_g, f'{source_name} (Grade)'))

            # Add markers at key points
            n_markers = min(8, len(cutoffs))
            marker_indices = np.linspace(0, len(cutoffs)-1, n_markers, dtype=int)
            self.ax_tonnage.scatter(
                cutoffs[marker_indices], tonnages[marker_indices],
                color=color, s=30, zorder=5, edgecolors='white', linewidths=0.8
            )

        # Styling
        self.ax_tonnage.set_facecolor(bg_color)
        self.ax_tonnage.set_xlabel('Cutoff Grade', fontsize=11, color=text_color, fontweight='medium')
        self.ax_tonnage.set_ylabel('Tonnage', fontsize=11, color='#58a6ff', fontweight='medium')
        self.ax_grade.set_ylabel('Average Grade', fontsize=11, color='#f0883e', fontweight='medium')
        self.ax_grade.yaxis.set_label_position('right')

        # Title
        self.ax_tonnage.set_title(
            'Grade-Tonnage Comparison', fontsize=13, color=text_color,
            fontweight='bold', pad=15
        )

        # Grid
        self.ax_tonnage.grid(True, alpha=0.15, color=grid_color, linestyle='-', linewidth=0.5)
        self.ax_tonnage.set_axisbelow(True)

        # Format tonnage axis
        def format_tonnage(x, p):
            if x >= 1e9:
                return f'{x/1e9:.1f}B'
            elif x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.0f}K'
            return f'{x:.0f}'

        self.ax_tonnage.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(format_tonnage)
        )

        # Tick styling
        self.ax_tonnage.tick_params(axis='y', colors='#58a6ff', labelsize=10)
        self.ax_tonnage.tick_params(axis='x', colors=text_color, labelsize=10)
        self.ax_grade.tick_params(axis='y', colors='#f0883e', labelsize=10)

        # Spine styling
        for spine in ['top']:
            self.ax_tonnage.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            self.ax_tonnage.spines[spine].set_color(grid_color)
            self.ax_tonnage.spines[spine].set_linewidth(0.8)
        self.ax_tonnage.spines['right'].set_visible(False)

        self.ax_grade.spines['right'].set_color('#f0883e')
        self.ax_grade.spines['right'].set_linewidth(1.5)
        for spine in ['top', 'left', 'bottom']:
            self.ax_grade.spines[spine].set_visible(False)

        # Legend with all sources
        lines = [l for l, _ in tonnage_lines]
        labels = [lbl for _, lbl in tonnage_lines]
        legend = self.ax_tonnage.legend(
            lines, labels, loc='upper right', frameon=True,
            facecolor='#161b22', edgecolor=grid_color,
            fontsize=9, labelcolor=text_color
        )
        legend.get_frame().set_alpha(0.9)

        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

        # Update statistics text with comparison
        self._update_comparison_stats()

    def _update_comparison_stats(self):
        """Update statistics text with comparison table."""
        if not self._comparison_results:
            return

        stats_lines = ["GRADE-TONNAGE COMPARISON SUMMARY", "=" * 50, ""]

        # Build comparison table
        header = f"{'Source':<30} {'Total Tonnage':>15} {'Avg Grade':>12}"
        stats_lines.append(header)
        stats_lines.append("-" * 60)

        for source_name, result in self._comparison_results.items():
            curve = result['curve']
            stats = curve.global_statistics
            total_tonnage = stats.get('total_tonnage', 0)
            grade_stats = stats.get('grade_statistics', {})
            mean_grade = grade_stats.get('mean', 0)

            # Format tonnage
            if total_tonnage >= 1e9:
                ton_str = f"{total_tonnage/1e9:.2f}B"
            elif total_tonnage >= 1e6:
                ton_str = f"{total_tonnage/1e6:.2f}M"
            else:
                ton_str = f"{total_tonnage:,.0f}"

            stats_lines.append(f"{source_name:<30} {ton_str:>15} {mean_grade:>12.3f}")

        stats_lines.append("")
        stats_lines.append("Note: Solid lines = Tonnage, Dashed lines = Grade")

        self.stats_text.setPlainText("\n".join(stats_lines))

    def _export_chart(self):
        """Export the chart as an image file."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'figure'):
            QMessageBox.warning(self, "Export Error", "No chart available to export.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Chart",
            "grade_tonnage_curve",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf);;All Files (*)"
        )

        if filename:
            try:
                # Ensure correct extension
                if not any(filename.lower().endswith(ext) for ext in ['.png', '.svg', '.pdf']):
                    if 'PNG' in selected_filter:
                        filename += '.png'
                    elif 'SVG' in selected_filter:
                        filename += '.svg'
                    elif 'PDF' in selected_filter:
                        filename += '.pdf'
                    else:
                        filename += '.png'

                # Export with high DPI for quality
                self.figure.savefig(
                    filename,
                    dpi=300,
                    bbox_inches='tight',
                    facecolor=self.figure.get_facecolor(),
                    edgecolor='none'
                )

                QMessageBox.information(self, "Export Complete", f"Chart exported to {filename}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors
        self.setStyleSheet(get_analysis_panel_stylesheet())
