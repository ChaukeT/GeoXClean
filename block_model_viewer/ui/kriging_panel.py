"""
3D Ordinary Kriging Panel

Provides UI for configuring and running 3D Ordinary Kriging via the controller.
Refactored for a cleaner, modern User Experience focused strictly on Ordinary Kriging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Optional dependencies
from .panel_manager import PanelCategory, DockArea
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    FigureCanvasQTAgg = None
    Figure = None

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QPushButton, QMessageBox,
    QTextEdit, QCheckBox, QFileDialog, QWidget, QSplitter,
    QScrollArea, QFrame, QDialog, QProgressBar, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from datetime import datetime

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import (
    get_grade_columns, validate_variable, populate_variable_combo,
    get_variable_from_combo_or_fallback
)
from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

# Import kriging export utilities
try:
    from ..models import kriging3d as krig
except ImportError:
    krig = None

logger = logging.getLogger(__name__)


def get_kriging_panel_stylesheet() -> str:
    """Enhanced stylesheet for Kriging panel with high contrast fixes."""
    colors = get_theme_colors()
    return f"""
        /* Ensure all text is bright white/light grey */
        QLabel {{
            color: {ModernColors.TEXT_PRIMARY};
            font-size: 10pt;
        }}

        /* Highlight the Refresh Banner */
        QLabel#NewDataBanner {{
            background-color: #1a3a5a;
            color: #4fc3f7;
            padding: 10px;
            border: 2px solid #2196F3;
            border-radius: 6px;
            font-weight: bold;
        }}

        /* Style Group Boxes to have a clear header */
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

        /* Make inputs pop */
        QDoubleSpinBox, QSpinBox, QComboBox {{
            background-color: #111;
            border: 1px solid #555;
            color: white;
            padding: 4px;
            min-height: 25px;
        }}

        QPushButton#RefreshBtn {{
            background-color: #2c3e50;
            border: 1px solid #3498db;
            color: #3498db;
            font-weight: bold;
            font-size: 14pt;
        }}
        QPushButton#RefreshBtn:hover {{
            background-color: #3498db;
            color: white;
        }}
    """


class KrigingPanel(BaseAnalysisPanel):
    """
    Panel for configuring and launching 3D Ordinary Kriging.
    Refactored for optimal UX/UI with split-pane layout.
    """
    # PanelManager metadata
    PANEL_ID = "KrigingPanel"
    PANEL_NAME = "Kriging Panel"
    PANEL_CATEGORY = PanelCategory.GEOSTATS
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    task_name = "kriging"
    request_visualization = pyqtSignal(dict)  # Signal to request visualization in main viewer
    progress_updated = pyqtSignal(int, str)

    def __init__(self, parent=None):
        # Initialize state BEFORE calling super().__init__
        self.drillhole_data: Optional[pd.DataFrame] = None
        self.variogram_results: Optional[Dict[str, Any]] = None
        self.kriging_results: Optional[Dict[str, Any]] = None
        self.transformation_metadata: Optional[Dict[str, Any]] = None
        self.registry = None
        
        # Pending payloads (if data loaded before UI ready)
        self._pending_drillhole_data = None
        self._pending_variogram_results = None
        self._ui_ready = False

        super().__init__(parent=parent, panel_id="kriging")

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        self.setWindowTitle("Ordinary Kriging")
        self.resize(1100, 750)

        # Build UI (required when using _build_ui pattern)
        self._build_ui()
        
        # UI is now built
        self._ui_ready = True

        self._init_registry_connections()

        # Connect progress signal to update method
        self.progress_updated.connect(self._update_progress)
        self._process_pending_data()
        
        # If we have drillhole data but UI wasn't ready when it was set, populate it now
        # (This handles the case where data was set before UI was built)
        if self.drillhole_data is not None and hasattr(self, 'variable_combo') and self.variable_combo.count() == 0:
            # Only populate if combo is empty (wasn't populated yet)
            numeric_cols = self.drillhole_data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH', 'HOLEID', 'GLOBAL_INTERVAL_ID',
                           # Compositing metadata columns
                           'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
                           'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO']
            numeric_cols = [c for c in numeric_cols if c.upper() not in exclude_cols]
            if numeric_cols:
                self.variable_combo.blockSignals(True)
                self.variable_combo.clear()
                self.variable_combo.addItems(sorted(numeric_cols))
                # Auto-select first variable if available
                if self.variable_combo.count() > 0:
                    self.variable_combo.setCurrentIndex(0)
                self.variable_combo.blockSignals(False)
                if hasattr(self, 'run_btn'):
                    self.run_btn.setEnabled(True)
                self._check_assisted_model()
    
    def _build_ui(self):
        """Build custom split-pane UI. Called by base class."""
        self._setup_ui()

    def _init_registry_connections(self) -> None:
        """Connect to the DataRegistry."""
        try:
            self.registry = self.get_registry()
            if not self.registry:
                return

            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            self.registry.variogramResultsLoaded.connect(self._on_variogram_results_loaded)
            if hasattr(self.registry, 'transformationMetadataLoaded'):
                self.registry.transformationMetadataLoaded.connect(self._on_transformation_metadata_loaded)
            
            # Listen for declustered data
            if hasattr(self.registry, 'declusteringResultsLoaded'):
                self.registry.declusteringResultsLoaded.connect(self._on_declustering_results_loaded)

            # Load initial state (prioritize declustered)
            # AUDIT FIX: Prefer get_estimation_ready_data for proper provenance
            declust = self.registry.get_data("declustering_results")
            if declust:
                self._on_declustering_results_loaded(declust)
            else:
                # Source-toggle panels must load full drillhole payload for proper source switching.
                existing_data = self.registry.get_drillhole_data()
                if existing_data:
                    self._on_drillhole_data_loaded(existing_data)
            
            existing_vario = self.registry.get_variogram_results()
            if existing_vario:
                self._on_variogram_results_loaded(existing_vario)
            
            # Retrieve transformation metadata if available
            existing_trans = self.registry.get_transformation_metadata() if hasattr(self.registry, 'get_transformation_metadata') else None
            if existing_trans:
                self.transformation_metadata = existing_trans

        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}")
            self.registry = None
    
    def _on_transformation_metadata_loaded(self, metadata):
        """Handle transformation metadata loaded from registry."""
        self.transformation_metadata = metadata

    def _process_pending_data(self):
        if self._pending_drillhole_data:
            self._on_drillhole_data_loaded(self._pending_drillhole_data)
            self._pending_drillhole_data = None
        if self._pending_variogram_results:
            self._on_variogram_results_loaded(self._pending_variogram_results)
            self._pending_variogram_results = None

    def _setup_ui(self):
        """Create a Split-Pane UI: Left (Config) / Right (Results)."""
        # Clear any existing layout from base class (BaseAnalysisPanel creates a scroll area)
        old_layout = self.layout()
        if old_layout:
            # Delete all child widgets
            while old_layout.count():
                item = old_layout.takeAt(0)
                if item:
                    widget = item.widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                    del item
            # Transfer layout ownership to temporary widget
            QWidget().setLayout(old_layout)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT PANEL: CONFIGURATION ---
        left_widget = QWidget()
        left_widget.setStyleSheet(get_kriging_panel_stylesheet())
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # Track if data has changed since last view
        self._pending_data_update = False

        # 1. DATA STATUS CARD (Fixed at top - always visible)
        status_card = QFrame()
        status_card.setStyleSheet("background-color: #1a1a1a; border-radius: 8px; border: 1px solid #333;")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 12, 12, 12)

        h_header = QHBoxLayout()
        header_title = QLabel("DATA SOURCE")
        header_title.setStyleSheet("color: #888; font-weight: bold; font-size: 9pt;")

        # THE REFRESH BUTTON (Large and Visible)
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setObjectName("RefreshBtn")
        self.refresh_btn.setFixedSize(40, 40)
        self.refresh_btn.setToolTip("Refresh data from registry (get latest composites/transforms)")
        self.refresh_btn.clicked.connect(self._manual_refresh)

        h_header.addWidget(header_title)
        h_header.addStretch()
        h_header.addWidget(self.refresh_btn)
        status_layout.addLayout(h_header)

        # Data source radio buttons (moved from data group)
        self.data_source_group = QButtonGroup()
        self.data_source_composited = QRadioButton("Composited Data")
        self.data_source_composited.setToolTip("Use composited drillhole data (recommended)")
        self.data_source_raw = QRadioButton("Raw Assay Data")
        self.data_source_raw.setToolTip("Use raw drillhole assay data")

        self.data_source_group.addButton(self.data_source_composited, 0)
        self.data_source_group.addButton(self.data_source_raw, 1)
        self.data_source_composited.setChecked(True)
        self.data_source_group.buttonClicked.connect(self._on_data_source_changed)

        status_layout.addWidget(self.data_source_composited)
        status_layout.addWidget(self.data_source_raw)

        # Data source status label (moved from data group)
        self.data_source_status_label = QLabel("Initializing data...")
        self.data_source_status_label.setWordWrap(True)
        self.data_source_status_label.setStyleSheet("font-size: 9pt; color: #aaa; margin-top: 8px;")
        status_layout.addWidget(self.data_source_status_label)

        left_layout.addWidget(status_card)

        # 2. CONFIGURATION SCROLL AREA
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 5, 0)
        self.scroll_layout.setSpacing(15)

        # 1. Data Selection (banner and variable selector only)
        self._create_data_group(self.scroll_layout)

        # 2. Variogram Model
        self._create_variogram_group(self.scroll_layout)

        # 3. Anisotropy
        self._create_anisotropy_group(self.scroll_layout)

        # 4. Grid & Search
        self._create_grid_search_group(self.scroll_layout)

        self.scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        left_layout.addWidget(scroll)
        
        # --- RIGHT PANEL: RESULTS & ACTIONS ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Progress Bar Section
        progress_group = QGroupBox("Progress")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; }")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #555;
                border-radius: 4px;
                background-color: {ModernColors.PANEL_BG};
                text-align: center;
                color: white;
                height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: #4CAF50;
                border-radius: 3px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet("color: #90a4ae; font-size: 10pt;")
        progress_layout.addWidget(self.progress_label)
        
        right_layout.addWidget(progress_group)

        # Event Log Area
        log_group = QGroupBox("Event Log")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{ 
                background-color: #2b2b2b; 
                color: {ModernColors.TEXT_PRIMARY}; 
                border: 1px solid #444;
                font-family: Consolas, monospace;
                font-size: 9pt;
            }}
        """)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        # Results Text Area
        results_group = QGroupBox("Estimation Results")
        results_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        res_layout = QVBoxLayout(results_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("Configure parameters and click 'Run Kriging'...")
        self.info_text.setStyleSheet(f"""
            QTextEdit {{ 
                background-color: #2b2b2b; 
                color: {ModernColors.TEXT_PRIMARY}; 
                border: 1px solid #444;
                font-family: Consolas, monospace;
            }}
        """)
        res_layout.addWidget(self.info_text)
        right_layout.addWidget(results_group, stretch=1)

        # Action Buttons
        self._create_action_buttons(right_layout)

        # Add to Splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)  # Config (smaller)
        splitter.setStretchFactor(1, 2)  # Results (larger)
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_data_group(self, layout):
        """Create data group - now only contains banner and variable selector.
        Data source selector moved to Data Status Card above."""
        group = QGroupBox("1. Input Data")
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #4fc3f7; border: 1px solid #444; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        vbox = QVBoxLayout(group)

        # New data notification banner (hidden by default) - Enhanced styling
        self._new_data_banner = QLabel("🔔 New data available! Click refresh (🔄) to load latest data/variogram.")
        self._new_data_banner.setObjectName("NewDataBanner")
        self._new_data_banner.setWordWrap(True)
        self._new_data_banner.setVisible(False)
        vbox.addWidget(self._new_data_banner)

        # Variable selector
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("Variable to Estimate:"))
        self.variable_combo = QComboBox()
        self.variable_combo.setMinimumWidth(200)
        hbox.addWidget(self.variable_combo)
        vbox.addLayout(hbox)

        layout.addWidget(group)

    def _create_variogram_group(self, layout):
        group = QGroupBox("2. Variogram Model")
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #444; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        vbox = QVBoxLayout(group)

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
        vbox.addLayout(btn_layout)

        self.assisted_model_label = QLabel("")
        self.assisted_model_label.setStyleSheet("color: #66bb6a; font-size: 10px;")
        vbox.addWidget(self.assisted_model_label)

        # Variogram signature lock indicator
        self.variogram_signature_label = QLabel("")
        self.variogram_signature_label.setStyleSheet("color: #90a4ae; font-size: 9px; font-family: monospace;")
        self.variogram_signature_label.setToolTip("Variogram signature for traceability (JORC/NI 43-101)")
        vbox.addWidget(self.variogram_signature_label)

        # Parameters Grid
        grid = QHBoxLayout()
        
        # Column 1
        col1 = QVBoxLayout()
        col1.addWidget(QLabel("Model Type:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Spherical", "Exponential", "Gaussian"])
        col1.addWidget(self.model_combo)
        
        col1.addWidget(QLabel("Nugget (C₀):"))
        self.nugget_spin = QDoubleSpinBox()
        self.nugget_spin.setRange(0.0, 10000.0)
        self.nugget_spin.setDecimals(3)
        col1.addWidget(self.nugget_spin)
        
        # Column 2
        col2 = QVBoxLayout()
        col2.addWidget(QLabel("Partial Sill (C):"))
        self.sill_spin = QDoubleSpinBox()
        self.sill_spin.setRange(0.001, 10000.0)
        self.sill_spin.setValue(1.0)
        self.sill_spin.setDecimals(3)
        col2.addWidget(self.sill_spin)
        
        col2.addWidget(QLabel("Range (a):"))
        self.range_spin = QDoubleSpinBox()
        self.range_spin.setRange(1.0, 50000.0)
        self.range_spin.setValue(100.0)
        self.range_spin.setSuffix(" m")
        col2.addWidget(self.range_spin)

        grid.addLayout(col1)
        grid.addSpacing(20)
        grid.addLayout(col2)
        vbox.addLayout(grid)
        
        layout.addWidget(group)

    def _create_anisotropy_group(self, layout):
        group = QGroupBox("3. Anisotropy (Optional)")
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #ba68c8; border: 1px solid #444; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        vbox = QVBoxLayout(group)
        
        grid = QHBoxLayout()
        
        # Azimuth
        l1 = QVBoxLayout()
        l1.addWidget(QLabel("Azimuth (Strike):"))
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0.0, 360.0)
        self.azimuth_spin.setSuffix("°")
        l1.addWidget(self.azimuth_spin)
        grid.addLayout(l1)
        
        # Dip
        l2 = QVBoxLayout()
        l2.addWidget(QLabel("Dip:"))
        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(-90.0, 90.0)
        self.dip_spin.setSuffix("°")
        l2.addWidget(self.dip_spin)
        grid.addLayout(l2)
        
        vbox.addLayout(grid)
        
        self.visualize_aniso_btn = QPushButton("Visualize Ellipsoid")
        self.visualize_aniso_btn.clicked.connect(self.visualize_anisotropy_ellipsoid)
        vbox.addWidget(self.visualize_aniso_btn)
        
        layout.addWidget(group)

    def _create_grid_search_group(self, layout):
        group = QGroupBox("4. Grid & Search")
        group.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #90a4ae; border: 1px solid #444; margin-top: 6px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
        """)
        vbox = QVBoxLayout(group)
        
        # Grid Origin
        origin_label = QLabel("Grid Origin (corner of first block):")
        origin_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        vbox.addWidget(origin_label)
        
        h0 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(1)
        self.xmin_spin.setValue(0)
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(1)
        self.ymin_spin.setValue(0)
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e9, 1e9)
        self.zmin_spin.setDecimals(1)
        self.zmin_spin.setValue(0)
        h0.addWidget(QLabel("X₀:"))
        h0.addWidget(self.xmin_spin)
        h0.addWidget(QLabel("Y₀:"))
        h0.addWidget(self.ymin_spin)
        h0.addWidget(QLabel("Z₀:"))
        h0.addWidget(self.zmin_spin)
        vbox.addLayout(h0)
        
        # Grid Spacing
        vbox.addWidget(QLabel("Block Size (Resolution):"))
        grid_hbox = QHBoxLayout()
        
        for axis, default in [("X", 10.0), ("Y", 10.0), ("Z", 5.0)]:
            sub = QHBoxLayout()
            sub.addWidget(QLabel(f"{axis}:"))
            spin = QDoubleSpinBox()
            spin.setRange(0.1, 1000.0)
            spin.setValue(default)
            setattr(self, f"grid_{axis.lower()}_spin", spin)
            sub.addWidget(spin)
            grid_hbox.addLayout(sub)
        vbox.addLayout(grid_hbox)
        
        # Auto-detect button
        auto_btn = QPushButton("Auto-Detect from Drillholes")
        auto_btn.setToolTip("Calculate grid origin and size to cover all drillhole data with padding")
        auto_btn.clicked.connect(self._auto_detect_grid)
        vbox.addWidget(auto_btn)
        
        # Auto-fit checkbox (enabled by default)
        self.auto_fit_grid_check = QCheckBox("Auto-fit grid when data loads")
        self.auto_fit_grid_check.setChecked(True)
        self.auto_fit_grid_check.setToolTip("Automatically restrict grid to drillhole extent when new data is loaded.\nPrevents estimation outside the data coverage area.")
        vbox.addWidget(self.auto_fit_grid_check)
        
        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")
        vbox.addWidget(line)
        
        # Search Params
        search_grid = QHBoxLayout()
        
        s1 = QVBoxLayout()
        s1.addWidget(QLabel("Neighbors:"))
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(4, 100)
        self.neighbors_spin.setValue(12)
        s1.addWidget(self.neighbors_spin)
        search_grid.addLayout(s1)
        
        s2 = QVBoxLayout()
        self.use_max_dist_check = QCheckBox("Limit Distance:")
        self.use_max_dist_check.stateChanged.connect(self._toggle_max_dist)
        s2.addWidget(self.use_max_dist_check)
        
        self.max_dist_spin = QDoubleSpinBox()
        self.max_dist_spin.setRange(10.0, 10000.0)
        self.max_dist_spin.setValue(200.0)
        self.max_dist_spin.setSuffix(" m")
        self.max_dist_spin.setEnabled(False)
        s2.addWidget(self.max_dist_spin)
        search_grid.addLayout(s2)
        
        vbox.addLayout(search_grid)

        # Professional Multi-Pass Search
        vbox.addWidget(QLabel(""))  # Spacer
        pro_label = QLabel("Professional Search Strategy:")
        pro_label.setStyleSheet("font-weight: bold; color: #66bb6a;")
        vbox.addWidget(pro_label)

        self.multi_pass_check = QCheckBox("Enable Multi-Pass Search (JORC/NI 43-101)")
        self.multi_pass_check.setToolTip(
            "Industry-standard 3-pass search strategy:\n"
            "• Pass 1 (Strict): min=8, max=12, ellipsoid=1.0×\n"
            "• Pass 2 (Relaxed): min=6, max=24, ellipsoid=1.5×\n"
            "• Pass 3 (Fallback): min=4, max=32, ellipsoid=2.0×\n\n"
            "Reduces unestimated blocks and provides audit trail."
        )
        self.multi_pass_check.setChecked(False)  # Opt-in
        self.multi_pass_check.stateChanged.connect(self._toggle_multi_pass)
        vbox.addWidget(self.multi_pass_check)

        # Professional defaults button
        self.use_pro_defaults_btn = QPushButton("Use Professional Defaults")
        self.use_pro_defaults_btn.setToolTip("Load JORC/NI 43-101 compliant 3-pass configuration")
        self.use_pro_defaults_btn.clicked.connect(self._load_professional_defaults)
        self.use_pro_defaults_btn.setEnabled(False)
        vbox.addWidget(self.use_pro_defaults_btn)

        # QA Metrics checkbox
        self.compute_qa_check = QCheckBox("Compute QA Metrics (KE, SoR, Negative Weights)")
        self.compute_qa_check.setToolTip(
            "Compute professional QA metrics:\n"
            "• Kriging Efficiency (KE)\n"
            "• Slope of Regression (SoR)\n"
            "• Percentage of Negative Weights\n"
            "• Pass Number\n"
            "• Distance to Nearest Sample\n"
            "• Number of Samples Used"
        )
        self.compute_qa_check.setChecked(True)  # Enabled by default for professional standard
        vbox.addWidget(self.compute_qa_check)

        layout.addWidget(group)

    def _toggle_max_dist(self, state):
        self.max_dist_spin.setEnabled(state == Qt.CheckState.Checked.value)

    def _toggle_multi_pass(self, state):
        """Enable/disable professional multi-pass controls."""
        enabled = state == Qt.CheckState.Checked.value
        self.use_pro_defaults_btn.setEnabled(enabled)
        if enabled:
            self._log_event("Multi-pass search enabled - use Professional Defaults or configure manually", "info")
        else:
            self._log_event("Multi-pass search disabled - using legacy single-pass mode", "warning")

    def _load_professional_defaults(self):
        """Load professional-standard 3-pass configuration."""
        try:
            from ..geostats.kriging_job_params import SearchConfig

            # Get base max distance
            base_max_dist = self.max_dist_spin.value() if self.use_max_dist_check.isChecked() else None

            # Create professional config
            search_config = SearchConfig.create_professional_default(base_max_distance=base_max_dist)

            # Store for later use
            self._professional_search_config = search_config

            # Update UI to show configuration loaded
            self._log_event("Professional defaults loaded: 3-pass search strategy", "success")
            self._log_event("  Pass 1: min=8, max=12, ellipsoid=1.0×", "info")
            self._log_event("  Pass 2: min=6, max=24, ellipsoid=1.5×", "info")
            self._log_event("  Pass 3: min=4, max=32, ellipsoid=2.0×", "info")

            # Show in info text as well
            if hasattr(self, 'info_text'):
                self.info_text.append(
                    "<br><span style='color: #66bb6a;'>✓ Professional 3-pass search loaded</span>"
                )
        except Exception as e:
            logger.error(f"Failed to load professional defaults: {e}", exc_info=True)
            QMessageBox.warning(self, "Configuration Error", f"Failed to load professional defaults:\n{e}")

    def _create_action_buttons(self, layout):
        # Primary Action
        self.run_btn = QPushButton("RUN ORDINARY KRIGING")
        self.run_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #43A047; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)

        # Secondary Actions
        grid = QHBoxLayout()
        
        self.visualize_btn = QPushButton("Visualize 3D")
        self.visualize_btn.clicked.connect(self.visualize_results)
        self.visualize_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)

        self.view_table_btn = QPushButton("View Table")
        self.view_table_btn.clicked.connect(self.open_results_table)
        self.view_table_btn.setEnabled(False)

        grid.addWidget(self.visualize_btn)
        grid.addWidget(self.export_btn)
        grid.addWidget(self.view_table_btn)
        layout.addLayout(grid)

        # Clear
        self.clear_btn = QPushButton("Clear Memory")
        self.clear_btn.setStyleSheet("background-color: #d32f2f; color: white; border: none; padding: 5px;")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setEnabled(False)
        layout.addWidget(self.clear_btn)

    # ------------------------------------------------------------------
    # Data Handling
    # ------------------------------------------------------------------

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_drillhole_data_loaded(data)
    
    def _on_declustering_results_loaded(self, results):
        """Handle new declustering results."""
        if not results or 'weighted_dataframe' not in results:
            return

        # Log receipt
        logger.info("Received declustered data for kriging analysis")

        # Store and use
        df = results['weighted_dataframe']

        # Ensure X, Y, Z columns exist
        df = ensure_xyz_columns(df)

        # AUDIT FIX: Set provenance attributes for JORC/SAMREC compliance
        df.attrs['source_type'] = 'declustered'
        df.attrs['lineage_gate_passed'] = True
        df.attrs['parent_source'] = 'composites'

        self.drillhole_data = df
        
        # Update UI
        if hasattr(self, 'data_source_status_label'):
            self.data_source_status_label.setText("Using: ✅ Declustered data")
            self.data_source_status_label.setStyleSheet("font-size: 9px; color: #4CAF50;")
            
        # Refresh variable combo
        if hasattr(self, 'variable_combo') and self.variable_combo is not None:
            populate_variable_combo(self.variable_combo, self.drillhole_data)
            
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(True)

    def _on_drillhole_data_loaded(self, data):
        """Handle drillhole data from registry.

        Shows notification banner if panel is visible, otherwise marks for refresh.
        """
        # Store data for later use
        self._pending_drillhole_data = data

        # If panel is visible, show notification banner (user decides when to refresh)
        if self.isVisible():
            if hasattr(self, '_new_data_banner'):
                self._new_data_banner.setVisible(True)
                self._pending_data_update = True
                logger.info("KrigingPanel: New drillhole data available, notification shown")
            return

        # Panel not visible - process immediately
        self._process_drillhole_data(data)

    def _manual_refresh(self):
        """Manual refresh - reload data from registry."""
        try:
            registry = self.get_registry()
            if registry:
                # Get latest drillhole data
                data = registry.get_drillhole_data()
                if data:
                    self._process_drillhole_data(data)

                # Also try to get latest variogram results
                vario = registry.get_data("variogram_results")
                if vario:
                    self.set_variogram_results(vario)
                    self.load_variogram_parameters()

                # Hide notification banner
                if hasattr(self, '_new_data_banner'):
                    self._new_data_banner.setVisible(False)
                self._pending_data_update = False

                # Show feedback
                if hasattr(self, 'info_text') and self.info_text is not None:
                    self.info_text.append("<br><span style='color: #81c784;'>✓ Data refreshed from registry.</span>")
                logger.info("KrigingPanel: Manual refresh completed")
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}", exc_info=True)
            if hasattr(self, 'info_text') and self.info_text is not None:
                self.info_text.append(f"<br><span style='color: #e57373;'>✗ Refresh failed: {e}</span>")

    def showEvent(self, event):
        """Auto-refresh when panel becomes visible."""
        super().showEvent(event)

        # If there's a pending update, apply it now
        if getattr(self, '_pending_data_update', False):
            if hasattr(self, '_pending_drillhole_data') and self._pending_drillhole_data is not None:
                self._process_drillhole_data(self._pending_drillhole_data)
                if hasattr(self, '_new_data_banner'):
                    self._new_data_banner.setVisible(False)
                self._pending_data_update = False

    def _process_drillhole_data(self, data):
        """
        Handle drillhole data loaded from registry.
        Accepts either a dict (with 'composites'/'assays' keys) or a DataFrame.
        Respects user's data source selection.
        """
        if not self._ui_ready:
            self._pending_drillhole_data = data
            return

        # Log diagnostic info about registry contents
        data_source = log_registry_data_status("Ordinary Kriging", data)
        
        # Store registry data
        self._registry_data = data
        
        # Extract DataFrame from dict if needed - RESPECT USER'S SELECTION
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
            composites_available = composites is not None and not (hasattr(composites, 'empty') and composites.empty)
            assays_available = assays is not None and not (hasattr(assays, 'empty') and assays.empty)
            
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
                    logger.warning(
                        "LINEAGE WARNING: Using raw assays for kriging. "
                        "Consider running compositing first for JORC/SAMREC compliance."
                    )
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
        else:
            logger.warning(f"Kriging panel: Unexpected data type: {type(data)}")
            return
        
        # Update radio button states
        if hasattr(self, 'data_source_composited') and hasattr(self, 'data_source_raw'):
            self.data_source_composited.setEnabled(composites_available)
            self.data_source_raw.setEnabled(assays_available)

            # Update status label with enhanced visibility
            if hasattr(self, 'data_source_status_label'):
                count = len(df) if df is not None else 0
                using_composites = df is composites if df is not None else False
                source_type = "Composites" if using_composites else "Raw Assays"

                if composites_available and assays_available:
                    self.data_source_status_label.setText(
                        f"Status: <b style='color:#4CAF50'>ACTIVE</b><br>"
                        f"Mode: {source_type}<br>"
                        f"Composites: {len(composites):,} | Assays: {len(assays):,}"
                    )
                    # Change refresh button border to green to indicate active data
                    if hasattr(self, 'refresh_btn'):
                        self.refresh_btn.setStyleSheet(
                            "QPushButton#RefreshBtn { background-color: #2c3e50; border: 2px solid #27ae60; color: #27ae60; font-weight: bold; font-size: 14pt; }"
                            "QPushButton#RefreshBtn:hover { background-color: #3498db; color: white; }"
                        )
                elif composites_available:
                    self.data_source_status_label.setText(
                        f"Status: <b style='color:#4CAF50'>ACTIVE</b><br>"
                        f"Mode: Composites<br>"
                        f"Samples: {len(composites):,}"
                    )
                    if hasattr(self, 'refresh_btn'):
                        self.refresh_btn.setStyleSheet(
                            "QPushButton#RefreshBtn { background-color: #2c3e50; border: 2px solid #27ae60; color: #27ae60; font-weight: bold; font-size: 14pt; }"
                            "QPushButton#RefreshBtn:hover { background-color: #3498db; color: white; }"
                        )
                elif assays_available:
                    self.data_source_status_label.setText(
                        f"Status: <b style='color:#FF9800'>ACTIVE</b><br>"
                        f"Mode: Raw Assays<br>"
                        f"Samples: {len(assays):,}"
                    )
                    if hasattr(self, 'refresh_btn'):
                        self.refresh_btn.setStyleSheet(
                            "QPushButton#RefreshBtn { background-color: #2c3e50; border: 2px solid #FF9800; color: #FF9800; font-weight: bold; font-size: 14pt; }"
                            "QPushButton#RefreshBtn:hover { background-color: #3498db; color: white; }"
                        )
                else:
                    self.data_source_status_label.setText(
                        f"Status: <b style='color:#e57373'>NO DATA</b><br>"
                        f"Click refresh to load data"
                    )
                    if hasattr(self, 'refresh_btn'):
                        self.refresh_btn.setStyleSheet(
                            "QPushButton#RefreshBtn { background-color: #2c3e50; border: 2px solid #e57373; color: #e57373; font-weight: bold; font-size: 14pt; }"
                            "QPushButton#RefreshBtn:hover { background-color: #3498db; color: white; }"
                        )
        
        if df is None or (hasattr(df, 'empty') and df.empty):
            logger.warning("Kriging panel: No valid drillhole data found")
            self.drillhole_data = None
            return
        
        # Ensure X, Y, Z columns exist
        df = ensure_xyz_columns(df)
        self.drillhole_data = df
        
        # Use centralized utility to populate variable combo
        grade_cols = get_grade_columns(df)
        
        # Only update UI if it's been built
        if hasattr(self, 'variable_combo') and self.variable_combo is not None:
            selected = populate_variable_combo(self.variable_combo, df)
            
            # Connect variable change
            try:
                self.variable_combo.currentTextChanged.disconnect()
            except TypeError:
                pass
            self.variable_combo.currentTextChanged.connect(self._check_assisted_model)
            
            if grade_cols:
                if hasattr(self, 'run_btn'):
                    self.run_btn.setEnabled(True)
                self._check_assisted_model()
                logger.info(f"Kriging panel: loaded {len(df)} samples, {len(grade_cols)} variables")
                
                # Auto-detect grid from drillhole extent if checkbox is enabled
                if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
                    try:
                        self._auto_detect_grid()
                        logger.info("Kriging panel: Auto-fitted grid to drillhole extent")
                    except Exception as e:
                        logger.warning(f"Kriging panel: Auto-fit grid failed: {e}")
            else:
                if hasattr(self, 'run_btn'):
                    self.run_btn.setEnabled(False)
        else:
            # UI not ready yet, store for later
            logger.debug("Kriging panel: UI not ready, data stored for later initialization")

    def set_drillhole_data(self, data: pd.DataFrame):
        """
        Legacy compatibility method - delegates to registry-based data loading.
        New code should use registry.drillholeDataLoaded signal.
        """
        # Delegate to registry-based method
        if data is not None:
            self._on_drillhole_data_loaded(data)

    def set_transformation_metadata(self, metadata: Dict):
        """Set transformation metadata for back-transformation of kriging results."""
        self.transformation_metadata = metadata
        logger.info(f"Stored transformation metadata for {len(metadata)} transformed columns")

    def _on_variogram_results_loaded(self, results: Dict[str, Any]):
        if not self._ui_ready:
            self._pending_variogram_results = results
            return
            
        self.set_variogram_results(results)
        if self.load_variogram_parameters():
            if hasattr(self, 'info_text') and self.info_text is not None:
                self.info_text.append("<br><span style='color: #81c784;'>✓ Variogram parameters auto-loaded.</span>")

    def set_variogram_results(self, results: Optional[Dict]):
        """Set variogram results for auto-parameter detection."""
        self.variogram_results = results
        # Only update UI if it's been built
        if hasattr(self, 'auto_vario_btn') and self.auto_vario_btn is not None:
            self.auto_vario_btn.setEnabled(results is not None)

    def load_variogram_parameters(self) -> bool:
        """Populate fields from variogram results.
        
        Priority order:
        1. combined_3d_model (what's displayed in variogram panel) - PREFERRED
        2. major variogram (for anisotropic models)
        3. omni variogram (fallback for isotropic)
        """
        if not self.variogram_results:
            return False
        
        # Only update UI if it's been built
        if not (hasattr(self, 'model_combo') and hasattr(self, 'range_spin') and 
                hasattr(self, 'sill_spin') and hasattr(self, 'nugget_spin')):
            logger.debug("Kriging panel: UI not ready for variogram parameter loading")
            return False
        
        try:
            model_type = self.model_combo.currentText().lower()
            
            # PRIORITY 1: Use combined_3d_model (matches what variogram panel displays)
            combined = self.variogram_results.get('combined_3d_model', {})
            if combined and combined.get('model_type', '').lower() == model_type:
                nugget = combined.get('nugget', 0.0)
                # SILL SEMANTICS: 'sill' is partial sill (C), 'total_sill' is C0+C
                partial_sill = combined.get('sill', 0.0)
                total_sill = combined.get('total_sill', nugget + partial_sill)
                major_range = combined.get('major_range', 100.0)
                
                self.range_spin.setValue(major_range)
                self.sill_spin.setValue(max(0.001, partial_sill))
                self.nugget_spin.setValue(nugget)

                logger.info(f"Loaded from combined_3d_model: range={major_range:.2f}m, partial_sill={partial_sill:.3f}, nugget={nugget:.3f}, total_sill={total_sill:.3f}")

                # Update variogram signature
                self._update_variogram_signature()

                return True
            
            # PRIORITY 2: Use major variogram (for anisotropic models)
            fitted = self.variogram_results.get('fitted_models', {})
            major = fitted.get('major', {}).get(model_type)
            if major:
                # Get total_sill (preferred) or calculate from nugget + sill
                total_sill = major.get('total_sill')
                if total_sill is None:
                    nugget = major.get('nugget', 0.0)
                    sill = major.get('sill', 0.0)
                    # Check if 'sill' is partial or total
                    if sill > nugget:
                        total_sill = nugget + sill  # Assume 'sill' is partial
                    else:
                        total_sill = sill  # Assume 'sill' is total
                
                nugget = major.get('nugget', 0.0)
                partial_sill = total_sill - nugget
                major_range = major.get('range', 100.0)
                
                self.range_spin.setValue(major_range)
                self.sill_spin.setValue(max(0.001, partial_sill))
                self.nugget_spin.setValue(nugget)

                logger.info(f"Loaded from major variogram: range={major_range:.2f}m, partial_sill={partial_sill:.3f}, nugget={nugget:.3f}")

                # Update variogram signature
                self._update_variogram_signature()

                return True
            
            # PRIORITY 3: Fallback to omni variogram (isotropic)
            omni = fitted.get('omni', {}).get(model_type)
            if omni:
                # Get total_sill (preferred) or calculate from nugget + sill
                total_sill = omni.get('total_sill')
                if total_sill is None:
                    nugget = omni.get('nugget', 0.0)
                    sill = omni.get('sill', 0.0)
                    # Check if 'sill' is partial or total
                    if sill > nugget:
                        total_sill = nugget + sill  # Assume 'sill' is partial
                    else:
                        total_sill = sill  # Assume 'sill' is total
                
                nugget = omni.get('nugget', 0.0)
                partial_sill = total_sill - nugget
                omni_range = omni.get('range', 100.0)
                
                # Try to match current model type, or fallback to first available
                if model_type not in fitted.get('omni', {}):
                    available_types = list(fitted.get('omni', {}).keys())
                    if available_types:
                        model_type = available_types[0]
                        idx = self.model_combo.findText(model_type.capitalize())
                        if idx >= 0:
                            self.model_combo.setCurrentIndex(idx)
                
                self.range_spin.setValue(omni_range)
                self.sill_spin.setValue(max(0.001, partial_sill))
                self.nugget_spin.setValue(nugget)

                logger.info(f"Loaded from omni variogram: range={omni_range:.2f}m, partial_sill={partial_sill:.3f}, nugget={nugget:.3f}")

                # Update variogram signature
                self._update_variogram_signature()

                return True
            
            logger.warning(f"No variogram parameters found for model type '{model_type}'")
            return False
            
        except Exception as e:
            logger.error(f"Error loading variogram: {e}", exc_info=True)
        return False

    def _update_variogram_signature(self):
        """Compute and display variogram signature for traceability."""
        try:
            from ..geostats.variogram_model import compute_variogram_signature

            # Gather current variogram parameters
            model_type = self.model_combo.currentText().lower()
            range_param = self.range_spin.value()
            sill = self.sill_spin.value()
            nugget = self.nugget_spin.value()
            sill_total = sill + nugget

            anisotropy = {
                'azimuth': self.azimuth_spin.value(),
                'dip': self.dip_spin.value(),
                'major_range': range_param,
                'minor_range': range_param,
                'vert_range': range_param,
            }

            # Extract anisotropic ranges if available
            if self.variogram_results:
                fitted_models = self.variogram_results.get('fitted_models', {})
                if 'major' in fitted_models and 'minor' in fitted_models:
                    major_models = fitted_models.get('major', {})
                    minor_models = fitted_models.get('minor', {})
                    if model_type in major_models and model_type in minor_models:
                        anisotropy['major_range'] = float(major_models[model_type].get('range', range_param))
                        anisotropy['minor_range'] = float(minor_models[model_type].get('range', range_param))
                    if 'vertical' in fitted_models and model_type in fitted_models['vertical']:
                        anisotropy['vert_range'] = float(fitted_models['vertical'][model_type].get('range', range_param))

            variogram_params = {
                'range': range_param,
                'sill': sill_total,
                'nugget': nugget,
                'model_type': model_type,
                'anisotropy': anisotropy
            }

            # Compute signature
            signature = compute_variogram_signature(variogram_params)

            # Display in UI
            if hasattr(self, 'variogram_signature_label'):
                self.variogram_signature_label.setText(f"🔒 Signature: {signature}")
                self.variogram_signature_label.setStyleSheet("color: #66bb6a; font-size: 9px; font-family: monospace;")

            logger.info(f"Variogram signature: {signature}")

        except Exception as e:
            logger.error(f"Failed to compute variogram signature: {e}", exc_info=True)
            if hasattr(self, 'variogram_signature_label'):
                self.variogram_signature_label.setText("")

    def _check_assisted_model(self):
        """Check for Variogram Assistant results."""
        # Only check if UI is ready
        if not (hasattr(self, 'variable_combo') and hasattr(self, 'use_assisted_btn') and 
                hasattr(self, 'assisted_model_label')):
            return
        
        has_model = False
        if self.controller and hasattr(self.controller, '_assisted_variogram_models'):
            var = self.variable_combo.currentText()
            if var in self.controller._assisted_variogram_models:
                has_model = True
                m_type = self.controller._assisted_variogram_models[var].get('model_type', 'N/A')
                self.assisted_model_label.setText(f"Available: {m_type}")
        
        self.use_assisted_btn.setEnabled(has_model)
        if not has_model:
            self.assisted_model_label.setText("")

    def _load_assisted_variogram(self):
        """Load variogram parameters from Variogram Assistant."""
        # Check if UI is ready
        if not (hasattr(self, 'variable_combo') and hasattr(self, 'model_combo') and
                hasattr(self, 'range_spin') and hasattr(self, 'sill_spin') and
                hasattr(self, 'nugget_spin') and hasattr(self, 'info_text')):
            QMessageBox.warning(self, "UI Not Ready", "Please wait for the panel to finish loading.")
            return
        
        if not self.controller or not hasattr(self.controller, '_assisted_variogram_models'):
            QMessageBox.warning(self, "No Model", "No assisted variogram model available.")
            return
            
        var = self.variable_combo.currentText()
        model = self.controller._assisted_variogram_models.get(var)
        
        if model:
            # Type
            raw_type = model.get('model_type', 'spherical')
            base_type = raw_type.split('+')[0] if '+' in raw_type else raw_type
            idx = self.model_combo.findText(base_type.capitalize())
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            
            # Params
            self.range_spin.setValue(float(model.get('ranges', [100.0])[0]))
            sills = model.get('sills', [1.0])
            self.sill_spin.setValue(float(sum(sills)))
            self.nugget_spin.setValue(float(model.get('nugget', 0.0)))
            
            self.info_text.append(f"<br><span style='color: #81c784;'>✓ Loaded assisted model for {var}.</span>")
            logger.info(f"Loaded assisted variogram for {var}")

    # ------------------------------------------------------------------
    # Analysis Execution
    # ------------------------------------------------------------------

    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI."""
        
        # 1. Resolve Data Source (FIXED LOGIC)
        data_to_use = self.drillhole_data
        
        # Fallback: Try registry if local data is missing
        # AUDIT FIX: Use get_estimation_ready_data() to ensure proper provenance tracking
        if data_to_use is None or (hasattr(data_to_use, 'empty') and data_to_use.empty):
            try:
                registry = self.get_registry()
                if registry:
                    # Prefer get_estimation_ready_data() for JORC/SAMREC compliance
                    # This returns declustered or composited data with proper source_type attrs
                    # HARD GATE: Use estimation_ready mode which filters out error rows
                    # This method will raise ValueError if validation not passed
                    try:
                        # First try to get composites (preferred for kriging)
                        data_to_use = registry.get_estimation_ready_data(
                            data_type="composites",
                            raise_on_block=True
                        )
                        if data_to_use is not None:
                            data_to_use.attrs['source_type'] = 'composites'
                            data_to_use.attrs['lineage_gate_passed'] = True
                            logger.info(
                                f"Retrieved estimation-ready COMPOSITES from registry: "
                                f"shape={data_to_use.shape}"
                            )
                    except ValueError as ve:
                        # If composites not available, try assays
                        logger.info(f"Composites not available: {ve}, trying assays...")
                        try:
                            data_to_use = registry.get_estimation_ready_data(
                                data_type="assays",
                                raise_on_block=True
                            )
                            if data_to_use is not None:
                                data_to_use.attrs['source_type'] = 'assays'
                                data_to_use.attrs['lineage_gate_passed'] = True
                                logger.warning(
                                    "LINEAGE WARNING: Using assays for kriging (no composites). "
                                    "Consider running compositing first for JORC/SAMREC compliance."
                                )
                                logger.info(
                                    f"Retrieved estimation-ready ASSAYS from registry: "
                                    f"shape={data_to_use.shape}"
                                )
                        except ValueError as ve2:
                            # Both failed - this is a hard stop
                            logger.error(f"ESTIMATION BLOCKED: {ve2}")
                            raise ValueError(
                                f"Cannot retrieve estimation-ready data:\n{ve2}\n\n"
                                "Run QC validation and apply cleaned data to registry first."
                            )
            except Exception as e:
                logger.warning(f"Failed to retrieve data from registry fallback: {e}")

        # Final validation of data source
        if data_to_use is None:
            raise ValueError("No drillhole data available. Please load drillhole/composite data first.")
        
        if not isinstance(data_to_use, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data_to_use)}")
        
        if hasattr(data_to_use, 'empty') and data_to_use.empty:
            raise ValueError("Drillhole data is empty. Please load drillhole/composite data with samples.")
        
        # Update local reference if successful
        self.drillhole_data = data_to_use

        # 2. Get Variable
        combo = getattr(self, 'variable_combo', None)
        result = get_variable_from_combo_or_fallback(
            combo, data_to_use, context="Ordinary Kriging"
        )
        if not result.is_valid:
            raise ValueError(result.error_message)
        variable = result.variable

        # 3. Create DataFrame Copy (The Critical Fix)
        # Ensure we are passing a concrete DataFrame, not a view or None
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        try:
            data_df = data_to_use.copy()
            # Explicitly preserve attrs (source_type, lineage_gate_passed, etc.)
            if hasattr(data_to_use, 'attrs') and data_to_use.attrs:
                data_df.attrs = data_to_use.attrs.copy()
            logger.info(f"Created data_df copy: shape={data_df.shape}, columns={list(data_df.columns)[:5]}...")
            logger.debug(f"Preserved attrs: {list(data_df.attrs.keys()) if hasattr(data_df, 'attrs') else 'none'}")
        except Exception as e:
            logger.error(f"Failed to copy drillhole data: {e}", exc_info=True)
            raise ValueError(f"Failed to copy drillhole data: {e}")

        if data_df is None:
            logger.error("Data copy resulted in None object")
            raise ValueError("Data copy resulted in None object.")
        
        if not isinstance(data_df, pd.DataFrame):
            logger.error(f"data_df is not a DataFrame: {type(data_df)}")
            raise ValueError(f"data_df is not a DataFrame: {type(data_df)}")
        
        logger.info(f"gather_parameters returning params with data_df shape: {data_df.shape}")

        # 4. Gather Variogram & Grid Params
        model_type = self.model_combo.currentText().lower()
        range_param = self.range_spin.value()
        sill = self.sill_spin.value()
        nugget = self.nugget_spin.value()
        
        sill_total = sill + nugget
        
        anisotropy = {
            'azimuth': self.azimuth_spin.value(),
            'dip': self.dip_spin.value(),
            'major_range': range_param,
            'minor_range': range_param,
            'vert_range': range_param,
        }
        
        if self.variogram_results:
            fitted_models = self.variogram_results.get('fitted_models', {})
            if 'major' in fitted_models and 'minor' in fitted_models:
                major_models = fitted_models.get('major', {})
                minor_models = fitted_models.get('minor', {})
                if model_type in major_models and model_type in minor_models:
                    anisotropy['major_range'] = float(major_models[model_type].get('range', range_param))
                    anisotropy['minor_range'] = float(minor_models[model_type].get('range', range_param))
                if 'vertical' in fitted_models and model_type in fitted_models['vertical']:
                    anisotropy['vert_range'] = float(fitted_models['vertical'][model_type].get('range', range_param))

        variogram_params = {
            'range': range_param,
            'sill': sill_total,
            'nugget': nugget,
            'anisotropy': anisotropy
        }

        if hasattr(self, 'xmin_spin'):
            grid_origin = (
                self.xmin_spin.value(),
                self.ymin_spin.value(),
                self.zmin_spin.value()
            )
        else:
            if all(col in data_to_use.columns for col in ['X', 'Y', 'Z']):
                grid_origin = (
                    float(data_to_use['X'].min()),
                    float(data_to_use['Y'].min()),
                    float(data_to_use['Z'].min())
                )
            else:
                grid_origin = (0.0, 0.0, 0.0)
        
        grid_spacing = (
            self.grid_x_spin.value(),
            self.grid_y_spin.value(),
            self.grid_z_spin.value()
        )
        
        n_neighbors = int(self.neighbors_spin.value())
        max_distance = float(self.max_dist_spin.value()) if self.use_max_dist_check.isChecked() else None

        # Professional search configuration
        search_passes = None
        compute_qa_metrics = False

        if hasattr(self, 'multi_pass_check') and self.multi_pass_check.isChecked():
            # Multi-pass enabled - use professional config if loaded
            if hasattr(self, '_professional_search_config') and self._professional_search_config:
                search_passes = [
                    {
                        'min_neighbors': p.min_neighbors,
                        'max_neighbors': p.max_neighbors,
                        'ellipsoid_multiplier': p.ellipsoid_multiplier
                    }
                    for p in self._professional_search_config.passes
                ]
            else:
                # Fallback to default 3-pass if not explicitly loaded
                search_passes = [
                    {'min_neighbors': 8, 'max_neighbors': 12, 'ellipsoid_multiplier': 1.0},
                    {'min_neighbors': 6, 'max_neighbors': 24, 'ellipsoid_multiplier': 1.5},
                    {'min_neighbors': 4, 'max_neighbors': 32, 'ellipsoid_multiplier': 2.0},
                ]

        if hasattr(self, 'compute_qa_check'):
            compute_qa_metrics = self.compute_qa_check.isChecked()

        # Return the params dict ensuring data_df is present
        return {
            "data_df": data_df,
            "variable": variable,
            "variogram_params": variogram_params,
            "grid_origin": grid_origin,
            "grid_spacing": grid_spacing,
            "n_neighbors": n_neighbors,
            "max_distance": max_distance,
            "model_type": model_type,
            "layer_name": f"OK_{variable}",
            # Professional parameters
            "search_passes": search_passes,
            "compute_qa_metrics": compute_qa_metrics,
        }

    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "Please load drillhole/composite data first.")
            return False
        
        # Use centralized variable validation
        combo = getattr(self, 'variable_combo', None)
        result = validate_variable(
            combo.currentText() if combo else None,
            self.drillhole_data,
            context="Ordinary Kriging"
        )
        
        if not result.is_valid:
            QMessageBox.warning(self, "Variable Error", result.error_message)
            return False
        
        # Check required coordinate columns
        variable = result.variable
        required_cols = ['X', 'Y', 'Z']
        missing_cols = [col for col in required_cols if col not in self.drillhole_data.columns]
        if missing_cols:
            QMessageBox.warning(self, "Missing Columns", f"Required columns missing: {', '.join(missing_cols)}")
            return False
        
        return True
    
    # ------------------------------------------------------------------
    # Progress and Logging Helpers
    # ------------------------------------------------------------------
    
    def _log_event(self, message: str, level: str = "info"):
        """Add timestamped event to the log with color coding."""
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "info": f"{ModernColors.TEXT_PRIMARY}",
            "success": "#81c784",
            "warning": "#ffb74d",
            "error": "#e57373",
            "progress": "#4fc3f7"
        }
        color = colors.get(level, f"{ModernColors.TEXT_PRIMARY}")
        
        formatted = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.log_text.append(formatted)
        
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
    
    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and label with percentage and message."""
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return
            
        percent = max(0, min(100, percent))
        
        # Ensure progress bar is visible
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        
        self.progress_bar.setValue(percent)
        if message:
            self.progress_bar.setFormat(f"{percent}% - {message}")
        else:
            self.progress_bar.setFormat(f"{percent}%")
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setText(message if message else f"{percent}% complete")
        
        # NOTE: processEvents() removed to prevent Qt painter reentrancy issues.
        # Progress updates via signals are sufficient for UI responsiveness.
        
        # Log milestones and important progress updates (throttled)
        # Log every 10% to avoid flooding the log
        should_log = (
            percent % 10 == 0 or 
            percent in [0, 100]
        )
        
        if should_log and message:
            self._log_event(f"Progress: {percent}% - {message}", "progress")
    
    def show_progress(self, message: str) -> None:
        """Override to use built-in progress bar."""
        # Ensure progress bar is visible
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setVisible(True)
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setVisible(True)
        
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "progress")
        if hasattr(self, 'run_btn') and self.run_btn:
            self.run_btn.setEnabled(False)
            self.run_btn.setText("Running...")
    
    def hide_progress(self) -> None:
        """Override to use built-in progress bar."""
        if hasattr(self, 'run_btn') and self.run_btn:
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Kriging")
    
    def _check_data_lineage(self) -> bool:
        """
        DOWNSTREAM SAFETY: Verify data lineage before kriging estimation.

        HARD GATE: Kriging WILL NOT proceed if validation has not passed.
        This is a JORC/SAMREC compliance requirement.

        Kriging requires properly prepared data:
        1. Validated data (ERROR rows excluded)
        2. Composited data (consistent sample support)
        3. Optionally declustered (for bias correction)

        Returns:
            True if data is acceptable for kriging
            False if estimation must be BLOCKED
        """
        registry = getattr(self, 'registry', None)
        if not registry:
            logger.error("LINEAGE: No registry available - cannot verify data lineage")
            QMessageBox.critical(
                self, "System Error",
                "Data registry not available. Cannot proceed with kriging."
            )
            return False

        # HARD GATE: Use the new require_validation_for_estimation() method
        allowed, message = registry.require_validation_for_estimation()

        if not allowed:
            logger.error(f"ESTIMATION BLOCKED: {message}")
            QMessageBox.critical(
                self, "Validation Required",
                f"Cannot run kriging:\n\n{message}\n\n"
                "Open the QC Window to validate your data and fix or exclude errors."
            )
            return False

        # Log validation status for audit trail
        validation_status = registry.get_drillholes_validation_status()
        if validation_status == "WARN":
            logger.info(
                "LINEAGE: Drillhole validation passed with warnings. "
                "Proceeding with kriging - error rows have been excluded."
            )

        # Check if using composited vs raw data
        if hasattr(self, 'data_source_raw') and self.data_source_raw.isChecked():
            try:
                dh_data = registry.get_drillhole_data(mode="validated", copy_data=False)
                if dh_data:
                    has_composites = isinstance(dh_data.get('composites'), pd.DataFrame) and not dh_data.get('composites').empty
                    if has_composites:
                        logger.warning(
                            "LINEAGE WARNING: Using raw assays for kriging when composites are available. "
                            "Raw assays have inconsistent sample support which violates change-of-support "
                            "principles. Consider using composites for JORC/SAMREC compliance."
                        )
            except Exception as e:
                logger.warning(f"Could not check composites availability: {e}")

        return True

    def run_analysis(self) -> None:
        """Override to pass progress callback for real-time updates."""
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
            return

        if not self.validate_inputs():
            return

        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return

        try:
            params = self.gather_parameters()
            # Verify data_df is in params before proceeding
            if 'data_df' not in params:
                logger.error("gather_parameters() did not return 'data_df' key")
                QMessageBox.warning(self, "Parameter Error", "Failed to prepare data for kriging. Missing data_df.")
                return
            if params.get('data_df') is None:
                logger.error("gather_parameters() returned None for 'data_df'")
                QMessageBox.warning(self, "Parameter Error", "Data preparation failed. data_df is None.")
                return
            logger.info(f"run_analysis: params keys={list(params.keys())}, data_df shape={params['data_df'].shape if hasattr(params['data_df'], 'shape') else 'N/A'}")
        except ValueError as e:
            logger.error(f"gather_parameters() raised ValueError: {e}", exc_info=True)
            QMessageBox.warning(self, "Parameter Error", str(e))
            return
        except Exception as e:
            logger.error(f"gather_parameters() raised unexpected exception: {e}", exc_info=True)
            QMessageBox.warning(self, "Parameter Error", f"Unexpected error: {e}")
            return

        self.show_progress("Starting kriging...")

        def progress_callback(percent: int, message: str):
            """Update progress bar from worker thread using signals."""
            # Emit signal to update UI from main thread
            self.progress_updated.emit(percent, message)

        try:
            self.controller.run_kriging(
                params=params,
                callback=self.handle_results,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Failed to dispatch kriging: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results from controller."""
        import pyvista as pv
        
        # Update progress
        self._update_progress(100, "Complete!")
        
        viz = payload.get("visualization", {})
        mesh = viz.get("mesh")
        property_name = payload.get("property_name", "Estimate")
        variance_property = payload.get("variance_property")
        metadata = payload.get("metadata", {})
        
        if mesh is None:
            self._log_event("ERROR: Kriging failed to return a mesh", "error")
            QMessageBox.critical(self, "Error", "Kriging failed to return a mesh.")
            return

        # Extract variable name
        variable = property_name.replace("_OK_est", "").replace("_est", "")
        self._log_event(f"✓ KRIGING COMPLETE", "success")

        # Extract QA metrics if available (professional mode)
        qa_summary = metadata.get('qa_summary')
        if qa_summary:
            self._display_qa_summary(qa_summary)

        # Get estimates and variances from mesh (copy to detach from VTK buffers)
        estimates = mesh[property_name] if property_name in mesh.array_names else None
        variances = mesh[variance_property] if variance_property and variance_property in mesh.array_names else None
        
        if estimates is None:
            QMessageBox.critical(self, "Error", f"Property '{property_name}' not found in mesh.")
            return
        
        # Convert to numpy and copy to ensure stable storage in registry
        if hasattr(estimates, 'numpy'):
            estimates = estimates.numpy()
        estimates = np.array(estimates, copy=True)

        if variances is not None:
            if hasattr(variances, 'numpy'):
                variances = variances.numpy()
            variances = np.array(variances, copy=True)
        
        # Extract grid coordinates
        if isinstance(mesh, pv.StructuredGrid):
            grid_x = np.array(mesh.x.reshape(mesh.dimensions, order='F'), copy=True)
            grid_y = np.array(mesh.y.reshape(mesh.dimensions, order='F'), copy=True)
            grid_z = np.array(mesh.z.reshape(mesh.dimensions, order='F'), copy=True)
        else:
            points = mesh.points
            grid_x = np.array(points[:, 0], copy=True)
            grid_y = np.array(points[:, 1], copy=True)
            grid_z = np.array(points[:, 2], copy=True)
        
        # Handle back-transformation
        estimates_back = None
        if self.transformation_metadata and variable in self.transformation_metadata:
            transform_info = self.transformation_metadata[variable]
            estimates_back = self._back_transform_values(estimates, transform_info)
            variable = transform_info.get('original_col_name', variable)
        
        estimates_to_store = estimates_back if estimates_back is not None else estimates
        
        # Basic sanity check
        est_finite = estimates_to_store[np.isfinite(estimates_to_store)]
        est_min = est_finite.min() if len(est_finite) else np.nan
        est_max = est_finite.max() if len(est_finite) else np.nan
        if np.isfinite(est_min) and np.isfinite(est_max):
            if max(abs(est_min), abs(est_max)) > 1e6:
                logger.warning(f"Kriging estimates exceed sanity threshold: min={est_min}, max={est_max}")
                self._log_event(
                    f"WARNING: Estimates extremely large (min={est_min:.3e}, max={est_max:.3e}). "
                    "Check variogram/parameters.", "warning"
                )

        # Store results
        self.kriging_results = {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'estimates': estimates_to_store,
            'estimates_transformed': estimates if estimates_back is not None else None,
            'variances': variances if variances is not None else np.full_like(estimates_to_store, np.nan),
            'variable': variable,
            'property_name': property_name,
            'variance_property': variance_property,
            'model_type': metadata.get('variogram_model', self.model_combo.currentText()),
            'back_transformed': estimates_back is not None,
            'metadata': metadata,
        }
        
        # Register results
        if self.registry:
            try:
                self.registry.register_kriging_results(self.kriging_results, source_panel="Ordinary Kriging")
                self._log_event("Results registered to data registry", "info")
            except Exception as e:
                logger.warning(f"Failed to register kriging results: {e}")

            # Also register the block model DataFrame for cross-sections and other panels
            try:
                # Create block model DataFrame from grid results
                coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

                # Build DataFrame dict starting with coordinates and core estimates
                df_dict = {
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                    f'{variable}_est': estimates_to_store.ravel(),
                    f'{variable}_var': variances.ravel() if variances is not None else np.full_like(estimates_to_store.ravel(), np.nan)
                }

                # Extract QA metrics from mesh if present (professional mode)
                qa_metric_names = ['kriging_efficiency', 'slope_of_regression', 'n_samples',
                                   'pass_number', 'distance_to_nearest', 'pct_negative_weights']
                for qa_name in qa_metric_names:
                    qa_prop_name = f'{variable}_OK_{qa_name}'
                    if qa_prop_name in mesh.array_names:
                        qa_values = mesh[qa_prop_name]
                        if hasattr(qa_values, 'numpy'):
                            qa_values = qa_values.numpy()
                        qa_values = np.array(qa_values, copy=True)
                        # Use shorter column names for DataFrame
                        col_name_map = {
                            'kriging_efficiency': f'{variable}_KE',
                            'slope_of_regression': f'{variable}_SoR',
                            'n_samples': f'{variable}_NSamples',
                            'pass_number': f'{variable}_Pass',
                            'distance_to_nearest': f'{variable}_MinDist',
                            'pct_negative_weights': f'{variable}_NegWt%'
                        }
                        df_dict[col_name_map.get(qa_name, qa_prop_name)] = qa_values.ravel()

                block_df = pd.DataFrame(df_dict).dropna()

                # Register the block model with FULL metadata for audit trail
                # (variogram signature, search strategy, QA summary, etc.)
                block_metadata = {
                    'variable': variable,
                    'method': 'ordinary_kriging',
                    'grid_size': (len(np.unique(grid_x)), len(np.unique(grid_y)), len(np.unique(grid_z))),
                    'n_blocks': len(block_df),
                    # Include full controller metadata for audit trail
                    'variogram_signature': metadata.get('variogram_signature'),
                    'variogram_params': metadata.get('variogram_params'),
                    'variogram_model': metadata.get('variogram_model'),
                    'search_strategy': metadata.get('search_strategy'),
                    'qa_summary': metadata.get('qa_summary'),
                    'timestamp': metadata.get('timestamp'),
                    'data_source_type': metadata.get('data_source_type'),
                    'samples_used': metadata.get('samples_used'),
                }

                self.registry.register_block_model_generated(
                    block_df,
                    source_panel="Ordinary Kriging",
                    metadata=block_metadata
                )
                self._log_event("Block model registered to data registry", "info")
            except Exception as e:
                logger.warning(f"Failed to register kriging block model: {e}")

        # Log details
        valid_est = estimates_to_store[~np.isnan(estimates_to_store)]
        self._log_event(f"  Variable: {variable}", "info")
        self._log_event(f"  Grid points: {len(estimates_to_store):,}", "info")
        self._log_event(f"  Valid estimates: {len(valid_est):,}", "info")
        self._log_event(f"  → Click 'Visualize 3D' to view results", "info")
        
        # Update UI
        self.visualize_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.view_table_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.run_btn.setEnabled(True)
        
        # Report
        valid_est = estimates_to_store[~np.isnan(estimates_to_store)]
        valid_var = variances[~np.isnan(variances)] if variances is not None else np.array([])
        
        summary = f"""
        <br><b style='color: #4CAF50;'>✓ Kriging Completed Successfully!</b><br>
        <table style='margin-top: 10px;'>
        <tr><td><b>Variable:</b></td><td>{variable}</td></tr>
        <tr><td><b>Model:</b></td><td>{self.kriging_results['model_type']}</td></tr>
        <tr><td><b>Grid Points:</b></td><td>{len(estimates_to_store):,}</td></tr>
        <tr><td><b>Valid Estimates:</b></td><td>{len(valid_est):,} ({100*len(valid_est)/len(estimates_to_store):.1f}%)</td></tr>
        </table>
        <br><b>Estimate Statistics:</b>
        <table>
        <tr><td>Mean:</td><td>{valid_est.mean():.4f}</td></tr>
        <tr><td>Min:</td><td>{valid_est.min():.4f}</td></tr>
        <tr><td>Max:</td><td>{valid_est.max():.4f}</td></tr>
        <tr><td>Std:</td><td>{valid_est.std():.4f}</td></tr>
        </table>
        """
        
        if len(valid_var) > 0:
            summary += f"""
        <br><b>Variance Statistics:</b>
        <table>
        <tr><td>Mean:</td><td>{valid_var.mean():.4f}</td></tr>
        <tr><td>Min:</td><td>{valid_var.min():.4f}</td></tr>
        <tr><td>Max:</td><td>{valid_var.max():.4f}</td></tr>
        </table>
        """
        
        # Add metadata section if available
        if metadata:
            summary += self._format_metadata_html(metadata)
        
        self.info_text.setHtml(summary)
        logger.info(f"Kriging completed: {len(valid_est)} valid estimates")

    def _format_metadata_html(self, metadata: Dict) -> str:
        """Format metadata for HTML display (variogram signature, search strategy, etc.)."""
        html = "<br><hr><br><b style='color: #66bb6a;'>📋 Estimation Metadata (Audit Trail)</b><br>"
        html += "<table style=f'margin-top: 10px; color: {ModernColors.TEXT_PRIMARY};'>"
        
        # Variogram signature
        if 'variogram_signature' in metadata:
            html += f"<tr><td><b>Variogram Signature:</b></td><td><code style='color: #4CAF50;'>🔒 {metadata['variogram_signature']}</code></td></tr>"
        
        # Variogram model
        if 'variogram_model' in metadata:
            html += f"<tr><td><b>Variogram Model:</b></td><td>{metadata['variogram_model']}</td></tr>"
        
        # Search strategy
        if 'search_strategy' in metadata:
            strat = metadata['search_strategy']
            mode = strat.get('mode', 'unknown')
            html += f"<tr><td><b>Search Mode:</b></td><td>{mode}</td></tr>"
            
            if mode == 'multi_pass' and 'passes' in strat:
                html += "<tr><td><b>Passes:</b></td><td>"
                for p in strat['passes']:
                    html += f"Pass {p['pass_number']}: min={p['min_neighbors']}, max={p['max_neighbors']}, ×{p['ellipsoid_multiplier']}<br>"
                html += "</td></tr>"
        
        # Timestamp
        if 'timestamp' in metadata:
            html += f"<tr><td><b>Generated:</b></td><td>{metadata['timestamp']}</td></tr>"
        
        # Data source
        if 'data_source_type' in metadata:
            html += f"<tr><td><b>Data Source:</b></td><td>{metadata['data_source_type']}</td></tr>"
        
        # Samples used
        if 'samples_used' in metadata:
            html += f"<tr><td><b>Samples Used:</b></td><td>{metadata['samples_used']:,}</td></tr>"
        
        html += "</table>"
        return html

    def _back_transform_values(self, transformed_values: np.ndarray, transform_info: Dict) -> np.ndarray:
        """Back-transform kriging estimates from transformed space to original space."""
        transform_type = transform_info.get('transform_type', '')
        original_values = transform_info.get('original_values')
        
        valid_mask = np.isfinite(transformed_values)
        back_transformed = np.full_like(transformed_values, np.nan)
        
        if not valid_mask.any():
            return back_transformed
        
        trans_valid = transformed_values[valid_mask]
        
        try:
            if "Normal Score" in transform_type or "Gaussian" in transform_type:
                if SCIPY_AVAILABLE and original_values is not None:
                    percentiles = stats.norm.cdf(trans_valid)
                    percentiles = np.clip(percentiles, 0.0, 1.0)
                    
                    orig_valid = original_values[~np.isnan(original_values)]
                    if len(orig_valid) > 0:
                        orig_sorted = np.sort(orig_valid)
                        indices = (percentiles * (len(orig_sorted) - 1)).astype(int)
                        indices = np.clip(indices, 0, len(orig_sorted) - 1)
                        back_transformed[valid_mask] = orig_sorted[indices]
                    else:
                        back_transformed[valid_mask] = trans_valid
                else:
                    back_transformed[valid_mask] = trans_valid
            
            elif "Log10" in transform_type:
                back_transformed[valid_mask] = np.power(10.0, trans_valid)
                if transform_info.get('add_const', False):
                    back_transformed[valid_mask] -= transform_info.get('const', 0.0)
            
            elif "Log" in transform_type and "10" not in transform_type:
                back_transformed[valid_mask] = np.exp(trans_valid)
                if transform_info.get('add_const', False):
                    back_transformed[valid_mask] -= transform_info.get('const', 0.0)
            
            elif "Square Root" in transform_type:
                back_transformed[valid_mask] = np.square(trans_valid)
                if transform_info.get('add_const', False):
                    back_transformed[valid_mask] -= transform_info.get('const', 0.0)
            
            elif "Box-Cox" in transform_type:
                lambda_val = transform_info.get('boxcox_lambda', 0.0)
                if lambda_val == 0:
                    back_transformed[valid_mask] = np.exp(trans_valid)
                else:
                    back_transformed[valid_mask] = np.power(lambda_val * trans_valid + 1.0, 1.0 / lambda_val)
                if transform_info.get('add_const', False):
                    back_transformed[valid_mask] -= transform_info.get('const', 0.0)
            
            else:
                back_transformed[valid_mask] = trans_valid
        
        except Exception as e:
            logger.error(f"Error during back-transformation: {e}")
            back_transformed[valid_mask] = trans_valid
        
        return back_transformed

    def _display_qa_summary(self, qa_summary: Dict):
        """Display professional QA metrics summary in the UI."""
        try:
            # Log to event log
            self._log_event("=" * 50, "info")
            self._log_event("PROFESSIONAL QA METRICS SUMMARY", "success")
            self._log_event("=" * 50, "info")

            # Multi-pass performance
            if 'pass_1_count' in qa_summary:
                total = qa_summary.get('pass_1_count', 0) + qa_summary.get('pass_2_count', 0) + qa_summary.get('pass_3_count', 0)
                self._log_event("Multi-Pass Search Performance:", "info")
                for pass_num in [1, 2, 3]:
                    count = qa_summary.get(f'pass_{pass_num}_count', 0)
                    pct = count * 100.0 / total if total > 0 else 0
                    self._log_event(f"  Pass {pass_num}: {count:,} blocks ({pct:.1f}%)", "info")

                unest = qa_summary.get('unestimated_count', 0)
                if unest > 0:
                    pct = unest * 100.0 / (total + unest) if (total + unest) > 0 else 0
                    self._log_event(f"  Unestimated: {unest:,} blocks ({pct:.1f}%)", "warning")

            # QA Metrics
            self._log_event("Quality Metrics:", "info")
            ke_mean = qa_summary.get('kriging_efficiency_mean', np.nan)
            ke_min = qa_summary.get('kriging_efficiency_min', np.nan)
            self._log_event(f"  Kriging Efficiency: mean={ke_mean:.3f}, min={ke_min:.3f}", "info")

            sor_mean = qa_summary.get('slope_of_regression_mean', np.nan)
            self._log_event(f"  Slope of Regression: mean={sor_mean:.3f}", "info")

            neg_wt_max = qa_summary.get('pct_negative_weights_max', np.nan)
            self._log_event(f"  Max Negative Weights: {neg_wt_max:.1f}%", "info")

            # Flag warnings
            if ke_min < 0.3:
                self._log_event(f"⚠ WARNING: Some blocks have low Kriging Efficiency (<0.3)", "warning")

            if sor_mean < 0.8 or sor_mean > 1.2:
                self._log_event(f"⚠ WARNING: Slope of Regression outside [0.8, 1.2]", "warning")

            if neg_wt_max > 20.0:
                self._log_event(f"⚠ WARNING: Some blocks have >20% negative weights", "warning")

            self._log_event("=" * 50, "info")

            # Display in info text as HTML table
            if hasattr(self, 'info_text'):
                html = "<br><div style=f'background-color: {ModernColors.CARD_BG}; padding: 10px; border: 1px solid #66bb6a; border-radius: 5px;'>"
                html += "<h3 style='color: #66bb6a;'>Professional QA Summary</h3>"

                # Multi-pass table
                if 'pass_1_count' in qa_summary:
                    html += "<table style=f'color: {ModernColors.TEXT_PRIMARY}; margin-bottom: 10px;'>"
                    html += "<tr><th colspan='2' style='color: #ffb74d;'>Multi-Pass Performance</th></tr>"
                    total = qa_summary.get('pass_1_count', 0) + qa_summary.get('pass_2_count', 0) + qa_summary.get('pass_3_count', 0)
                    for pass_num in [1, 2, 3]:
                        count = qa_summary.get(f'pass_{pass_num}_count', 0)
                        pct = count * 100.0 / total if total > 0 else 0
                        html += f"<tr><td>Pass {pass_num}:</td><td>{count:,} ({pct:.1f}%)</td></tr>"
                    unest = qa_summary.get('unestimated_count', 0)
                    if unest > 0:
                        html += f"<tr style='color: #ff9800;'><td>Unestimated:</td><td>{unest:,}</td></tr>"
                    html += "</table>"

                # QA metrics table
                html += "<table style=f'color: {ModernColors.TEXT_PRIMARY};'>"
                html += "<tr><th colspan='2' style='color: #ffb74d;'>Quality Metrics</th></tr>"
                html += f"<tr><td>Kriging Efficiency (mean):</td><td>{ke_mean:.3f}</td></tr>"
                html += f"<tr><td>Kriging Efficiency (min):</td><td>{ke_min:.3f}</td></tr>"
                html += f"<tr><td>Slope of Regression (mean):</td><td>{sor_mean:.3f}</td></tr>"
                html += f"<tr><td>Max Negative Weights:</td><td>{neg_wt_max:.1f}%</td></tr>"
                html += "</table>"

                # Warnings
                warnings = []
                if ke_min < 0.3:
                    warnings.append("Low Kriging Efficiency detected")
                if sor_mean < 0.8 or sor_mean > 1.2:
                    warnings.append("Slope of Regression outside acceptable range")
                if neg_wt_max > 20.0:
                    warnings.append("High negative weights detected")

                if warnings:
                    html += "<div style='margin-top: 10px; color: #ff9800;'>"
                    html += "<strong>⚠ Warnings:</strong><ul>"
                    for w in warnings:
                        html += f"<li>{w}</li>"
                    html += "</ul></div>"

                html += "</div>"
                self.info_text.append(html)

        except Exception as e:
            logger.error(f"Failed to display QA summary: {e}", exc_info=True)
            self._log_event(f"Error displaying QA summary: {e}", "error")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def visualize_results(self):
        """Visualize kriging results in 3D."""
        if self.kriging_results is None:
            self._log_event("No results to visualize", "warning")
            QMessageBox.warning(self, "No Results", "Please run kriging first.")
            return
        
        self._log_event("Sending results to 3D viewer...", "progress")
        
        # Try signal-based approach first
        n_receivers = self.receivers(self.request_visualization)
        if n_receivers > 0:
            try:
                self.request_visualization.emit(self.kriging_results)
                self._log_event("✓ Results sent to 3D viewer", "success")
                return
            except Exception as e:
                logger.error(f"Signal emission error: {e}", exc_info=True)
        
        # Fallback to parent method
        if self.parent() and hasattr(self.parent(), 'visualize_kriging_results'):
            try:
                self.parent().visualize_kriging_results(self.kriging_results)
                self._log_event("✓ Results sent to 3D viewer", "success")
            except Exception as e:
                logger.error(f"Visualization error: {e}", exc_info=True)
                self._log_event(f"ERROR: {str(e)}", "error")
                QMessageBox.critical(self, "Visualization Error", f"Error visualizing results:\n{str(e)}")
        else:
            self._log_event("Cannot access main viewer", "warning")
            QMessageBox.information(self, "Info", "Cannot access main viewer for visualization.")

    def export_results(self):
        """Export kriging results to file."""
        if self.kriging_results is None:
            QMessageBox.warning(self, "No Results", "Please run kriging first.")
            return
        
        # Ask for export format
        reply = QMessageBox.question(
            self,
            "Export Format",
            "Choose export format:\n\nYes = CSV (point cloud)\nNo = VTK (structured grid)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Cancel:
            return
        
        format_type = 'csv' if reply == QMessageBox.StandardButton.Yes else 'vtk'
        extension = 'csv' if format_type == 'csv' else 'vts'
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Kriging Results",
            f"kriging_results.{extension}",
            f"{extension.upper()} Files (*.{extension})"
        )
        
        if not file_path:
            return
        
        if krig is None:
            QMessageBox.critical(self, "Export Error", "Kriging export module not available.")
            return
        
        try:
            krig.export_kriging_results(
                file_path,
                self.kriging_results['grid_x'],
                self.kriging_results['grid_y'],
                self.kriging_results['grid_z'],
                self.kriging_results['estimates'],
                self.kriging_results['variances'],
                format=format_type
            )
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"✓ Kriging results exported successfully!\n\nFile: {file_path}\nFormat: {format_type.upper()}"
            )
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Error exporting results:\n{str(e)}")

    def open_results_table(self):
        """Open kriging results as a table."""
        if self.kriging_results is None:
            QMessageBox.information(self, "No Results", "Please run kriging first.")
            return
        
        try:
            variable = self.kriging_results.get('variable', 'Estimate')
            
            # Get the registered block model DataFrame (includes QA metrics if present)
            df = None
            if hasattr(self, 'registry') and self.registry:
                try:
                    block_model = self.registry.get_block_model(copy_data=True)
                    if block_model is not None and isinstance(block_model, pd.DataFrame):
                        df = block_model
                        logger.info(f"Retrieved registered block model with {len(df.columns)} columns: {list(df.columns)}")
                    else:
                        logger.info("Registry block model is None or not a DataFrame")
                except Exception as e:
                    logger.warning(f"Could not retrieve registered block model: {e}")
            
            # Fallback: create DataFrame from kriging_results (legacy, no QA metrics)
            if df is None:
                logger.info("Fallback: Creating DataFrame from kriging_results (no QA metrics)")
                grid_x = self.kriging_results['grid_x']
                grid_y = self.kriging_results['grid_y']
                grid_z = self.kriging_results['grid_z']
                estimates = self.kriging_results['estimates']
                variances = self.kriging_results['variances']

                estimates = np.asarray(estimates)
                variances = np.asarray(variances)

                coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
                df = pd.DataFrame({
                    'X': coords[:, 0],
                    'Y': coords[:, 1],
                    'Z': coords[:, 2],
                    f'{variable} (Estimate)': estimates.ravel(),
                    'Variance': variances.ravel()
                }).dropna()

            title = f"Kriging Results - {variable}"
            
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
                layout.setContentsMargins(0, 0, 0, 0)
                
                panel = TableViewerPanel()
                layout.addWidget(panel)
                panel.set_dataframe(df, title=title)
                
                dialog.show()
                logger.info(f"Opened Table Viewer window for: {title}")
                
        except Exception as e:
            logger.error(f"Failed to open results table: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to open table view:\n{e}")

    def clear_results(self):
        """Clear kriging results from memory."""
        reply = QMessageBox.question(
            self,
            "Clear Results",
            "Are you sure you want to clear kriging results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.kriging_results = None
            self.info_text.clear()
            self.info_text.setPlaceholderText("Configure parameters and click 'Run Kriging'...")
            self.visualize_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.view_table_btn.setEnabled(False)
            self.clear_btn.setEnabled(False)
            logger.info("Kriging results cleared")

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults (no confirmation)."""
        # Clear internal state
        self.drillhole_data = None
        self.variogram_results = None
        self.kriging_results = None
        self.transformation_metadata = None

        # Clear results display
        if hasattr(self, 'info_text') and self.info_text:
            self.info_text.clear()
            self.info_text.setPlaceholderText("Configure parameters and click 'Run Kriging'...")

        # Disable result buttons
        for btn_name in ['visualize_btn', 'export_btn', 'view_table_btn', 'clear_btn']:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.setEnabled(False)

        # Clear matplotlib canvas if exists
        if hasattr(self, 'canvas') and self.canvas:
            try:
                self.canvas.figure.clear()
                self.canvas.draw()
            except Exception as e:
                logger.error(f"Failed to clear/draw matplotlib canvas: {e}", exc_info=True)

        # Call base class to clear common widgets
        super().clear_panel()
        logger.info("KrigingPanel: Panel fully cleared")

    def visualize_anisotropy_ellipsoid(self):
        """Visualize anisotropy ellipsoid in 3D viewer."""
        if self.variogram_results is None:
            QMessageBox.warning(
                self,
                "No Variogram Data",
                "Please load variogram results first.\nAnisotropy visualization requires directional variogram models."
            )
            return
        
        try:
            import pyvista as pv
            
            fitted_models = self.variogram_results.get('fitted_models', {})
            model_type = self.model_combo.currentText().lower()
            
            # Get ranges from directional variograms
            range_param = self.range_spin.value()
            major_range = range_param
            minor_range = range_param
            vert_range = range_param
            
            if 'major' in fitted_models and model_type in fitted_models['major']:
                major_range = fitted_models['major'][model_type].get('range', range_param)
            if 'minor' in fitted_models and model_type in fitted_models['minor']:
                minor_range = fitted_models['minor'][model_type].get('range', range_param)
            if 'vertical' in fitted_models and model_type in fitted_models['vertical']:
                vert_range = fitted_models['vertical'][model_type].get('range', range_param)
            
            # Get user-specified orientation
            azimuth = float(self.azimuth_spin.value())
            dip = float(self.dip_spin.value())
            
            # Calculate center point
            center = np.array([0.0, 0.0, 0.0])
            if self.drillhole_data is not None and all(col in self.drillhole_data.columns for col in ['X', 'Y', 'Z']):
                coords = self.drillhole_data[['X', 'Y', 'Z']].values
                valid_coords = coords[np.isfinite(coords).all(axis=1)]
                if len(valid_coords) > 0:
                    center = valid_coords.mean(axis=0)
            
            # Create ellipsoid
            ellipsoid = pv.Sphere(radius=1.0, center=[0, 0, 0], theta_resolution=30, phi_resolution=30)
            
            # Scale by ranges
            scale_transform = np.array([
                [major_range, 0, 0, 0],
                [0, minor_range, 0, 0],
                [0, 0, vert_range, 0],
                [0, 0, 0, 1]
            ], dtype=float)
            ellipsoid.transform(scale_transform)
            
            # Rotation matrices
            az_rad = np.deg2rad(azimuth)
            dip_rad = np.deg2rad(dip)
            
            Rz = np.array([
                [np.cos(az_rad), -np.sin(az_rad), 0, 0],
                [np.sin(az_rad), np.cos(az_rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float)
            
            Rx = np.array([
                [1, 0, 0, 0],
                [0, np.cos(dip_rad), -np.sin(dip_rad), 0],
                [0, np.sin(dip_rad), np.cos(dip_rad), 0],
                [0, 0, 0, 1]
            ], dtype=float)
            
            ellipsoid.transform(Rz @ Rx)
            
            # Translate to center
            translate_transform = np.eye(4, dtype=float)
            translate_transform[:3, 3] = center
            ellipsoid.transform(translate_transform)
            
            # Add to viewer
            if self.parent() and hasattr(self.parent(), 'viewer_widget'):
                renderer = self.parent().viewer_widget.renderer
                if renderer and renderer.plotter:
                    renderer.plotter.add_mesh(
                        ellipsoid,
                        style='wireframe',
                        color='red',
                        line_width=3,
                        opacity=0.8,
                        name='Anisotropy Ellipsoid',
                        pickable=False  # CRITICAL: Don't block mouse interactions with 3D viewer
                    )
                    renderer.plotter.render()
                    
                    QMessageBox.information(
                        self,
                        "Anisotropy Ellipsoid",
                        f"Ellipsoid displayed!\n\nAzimuth: {azimuth:.1f}°\nDip: {dip:.1f}°\n"
                        f"Major: {major_range:.1f}m\nMinor: {minor_range:.1f}m\nVertical: {vert_range:.1f}m"
                    )
                else:
                    QMessageBox.warning(self, "Error", "3D renderer not available.")
            else:
                QMessageBox.warning(self, "Error", "Cannot access 3D viewer.")
                
        except Exception as e:
            logger.error(f"Error visualizing anisotropy: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error creating visualization:\n{str(e)}")

    def get_kriging_results(self) -> Optional[Dict]:
        """Get stored kriging results."""
        return self.kriging_results

    def closeEvent(self, event):
        """Handle panel close - hide instead of close to preserve results."""
        event.ignore()
        self.hide()
        logger.info("Kriging panel hidden (results preserved)")
    
    def _auto_detect_grid(self):
        """
        Auto-detect grid parameters from drillhole coordinates.
        Uses the actual drillhole data extent to define a grid that covers all drillhole sample locations with padding.
        CRITICAL FIX: Prefer actual rendered drillhole bounds over DataFrame bounds to ensure grid aligns with what's visually displayed.
        """
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "No drillhole data loaded for grid detection.")
            return
        
        # Try to get actual rendered drillhole bounds first (most accurate)
        rendered_bounds = None
        renderer = self._find_renderer()
        
        if renderer:
            logger.info(f"Auto-detect: Found renderer of type {type(renderer).__name__}")
            has_cache = hasattr(renderer, '_drillhole_polylines_cache')
            cache = getattr(renderer, '_drillhole_polylines_cache', None) if has_cache else None
            
            if cache and 'hole_polys' in cache:
                try:
                    hole_polys = cache['hole_polys']
                    all_points = []
                    for hole_id, poly in hole_polys.items():
                        if hasattr(poly, 'points') and poly.n_points > 0:
                            all_points.append(poly.points)
                    
                    if all_points:
                        stacked_points = np.vstack(all_points)
                        rendered_bounds = {
                            'x_min': float(stacked_points[:, 0].min()),
                            'x_max': float(stacked_points[:, 0].max()),
                            'y_min': float(stacked_points[:, 1].min()),
                            'y_max': float(stacked_points[:, 1].max()),
                            'z_min': float(stacked_points[:, 2].min()),
                            'z_max': float(stacked_points[:, 2].max()),
                        }
                        logger.info("Auto-detect: ✓ Using RENDERED drillhole bounds")
                except Exception as e:
                    logger.warning(f"Could not extract rendered drillhole bounds: {e}", exc_info=True)
        
        # Use rendered bounds if available, otherwise fall back to DataFrame
        if rendered_bounds:
            x_min = rendered_bounds['x_min']
            x_max = rendered_bounds['x_max']
            y_min = rendered_bounds['y_min']
            y_max = rendered_bounds['y_max']
            z_min = rendered_bounds['z_min']
            z_max = rendered_bounds['z_max']
        else:
            df = self.drillhole_data
            if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
                QMessageBox.warning(self, "Missing Coordinates", "Drillhole data must have X, Y, Z columns for grid detection.")
                return
            
            valid_mask = df['X'].notna() & df['Y'].notna() & df['Z'].notna()
            valid_df = df[valid_mask]
            
            if valid_df.empty:
                QMessageBox.warning(self, "No Valid Data", "No valid coordinate data found in drillholes.")
                return
            
            x_min = float(valid_df['X'].min())
            x_max = float(valid_df['X'].max())
            y_min = float(valid_df['Y'].min())
            y_max = float(valid_df['Y'].max())
            z_min = float(valid_df['Z'].min())
            z_max = float(valid_df['Z'].max())
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Ensure non-zero ranges
        if x_range < 1e-6:
            x_range = 100.0
        if y_range < 1e-6:
            y_range = 100.0
        if z_range < 1e-6:
            z_range = 50.0
        
        # Use current block size values as defaults, or sensible mining defaults
        dx = self.grid_x_spin.value() if self.grid_x_spin.value() > 0.1 else 10.0
        dy = self.grid_y_spin.value() if self.grid_y_spin.value() > 0.1 else 10.0
        dz = self.grid_z_spin.value() if self.grid_z_spin.value() > 0.1 else 5.0
        
        # Add padding: 1 block on each side minimum, or 5% of range
        x_pad = max(dx, x_range * 0.05)
        y_pad = max(dy, y_range * 0.05)
        z_pad = max(dz, z_range * 0.05)
        
        # Calculate grid origin (snap to block size for cleaner coordinates)
        xmin = np.floor((x_min - x_pad) / dx) * dx
        ymin = np.floor((y_min - y_pad) / dy) * dy
        zmin = np.floor((z_min - z_pad) / dz) * dz
        
        # Set the UI values
        if hasattr(self, 'xmin_spin'):
            self.xmin_spin.setValue(xmin)
            self.ymin_spin.setValue(ymin)
            self.zmin_spin.setValue(zmin)
        
        logger.info(
            f"Auto-detected grid origin: ({xmin:.1f}, {ymin:.1f}, {zmin:.1f}), "
            f"spacing=({dx:.1f}, {dy:.1f}, {dz:.1f})m"
        )
        
        if hasattr(self, 'log_text'):
            self._log_event(
                f"Grid Auto-Detection: origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f}), spacing=({dx:.1f}, {dy:.1f}, {dz:.1f})m",
                "success"
            )
    
    def _find_renderer(self):
        """Find the renderer from various sources."""
        renderer = None
        
        # Method 0: Direct main_window reference
        if hasattr(self, 'main_window') and self.main_window is not None:
            if hasattr(self.main_window, 'viewer_widget'):
                renderer = getattr(self.main_window.viewer_widget, 'renderer', None)
                if renderer:
                    return renderer
        
        # Method 1: From controller
        if self.controller:
            renderer = getattr(self.controller, 'r', None)
            if renderer:
                return renderer
        
        # Method 2: From parent
        try:
            parent = self.parent()
            if parent and hasattr(parent, 'viewer_widget'):
                renderer = getattr(parent.viewer_widget, 'renderer', None)
                if renderer:
                    return renderer
        except Exception as e:
            logger.error(f"Failed to get renderer from parent: {e}", exc_info=True)
        
        # Method 3: Walk up parent chain
        try:
            widget = self
            for i in range(10):
                widget = widget.parent() if hasattr(widget, 'parent') else None
                if widget is None:
                    break
                if hasattr(widget, 'viewer_widget'):
                    renderer = getattr(widget.viewer_widget, 'renderer', None)
                    if renderer:
                        return renderer
        except Exception as e:
            logger.error(f"Failed to walk parent chain for renderer: {e}", exc_info=True)
        
        # Method 4: Try QApplication
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for window in app.topLevelWidgets():
                    if hasattr(window, 'viewer_widget'):
                        renderer = getattr(window.viewer_widget, 'renderer', None)
                        if renderer:
                            return renderer
        except Exception as e:
            logger.error(f"Failed to get renderer from QApplication widgets: {e}", exc_info=True)
        
        return None

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save.
        
        Returns:
            Dictionary of all user-configurable settings, or None if no settings.
        """
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Data selection
            settings['variable'] = get_safe_widget_value(self, 'variable_combo')
            
            # Data source
            if hasattr(self, 'data_source_composited'):
                settings['data_source'] = 'composited' if self.data_source_composited.isChecked() else 'raw'
            
            # Variogram model
            settings['model_type'] = get_safe_widget_value(self, 'model_combo')
            settings['nugget'] = get_safe_widget_value(self, 'nugget_spin')
            settings['sill'] = get_safe_widget_value(self, 'sill_spin')
            settings['range'] = get_safe_widget_value(self, 'range_spin')
            
            # Anisotropy
            settings['azimuth'] = get_safe_widget_value(self, 'azimuth_spin')
            settings['dip'] = get_safe_widget_value(self, 'dip_spin')
            
            # Grid origin
            settings['xmin'] = get_safe_widget_value(self, 'xmin_spin')
            settings['ymin'] = get_safe_widget_value(self, 'ymin_spin')
            settings['zmin'] = get_safe_widget_value(self, 'zmin_spin')
            
            # Grid spacing
            settings['grid_x'] = get_safe_widget_value(self, 'grid_x_spin')
            settings['grid_y'] = get_safe_widget_value(self, 'grid_y_spin')
            settings['grid_z'] = get_safe_widget_value(self, 'grid_z_spin')
            
            # Search settings
            settings['neighbors'] = get_safe_widget_value(self, 'neighbors_spin')
            settings['use_max_dist'] = get_safe_widget_value(self, 'use_max_dist_check')
            settings['max_dist'] = get_safe_widget_value(self, 'max_dist_spin')
            settings['auto_fit_grid'] = get_safe_widget_value(self, 'auto_fit_grid_check')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save kriging panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load.
        
        Args:
            settings: Dictionary of previously saved settings
        """
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Data selection
            set_safe_widget_value(self, 'variable_combo', settings.get('variable'))
            
            # Data source
            if 'data_source' in settings:
                if settings['data_source'] == 'composited' and hasattr(self, 'data_source_composited'):
                    self.data_source_composited.setChecked(True)
                elif settings['data_source'] == 'raw' and hasattr(self, 'data_source_raw'):
                    self.data_source_raw.setChecked(True)
            
            # Variogram model
            set_safe_widget_value(self, 'model_combo', settings.get('model_type'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget'))
            set_safe_widget_value(self, 'sill_spin', settings.get('sill'))
            set_safe_widget_value(self, 'range_spin', settings.get('range'))
            
            # Anisotropy
            set_safe_widget_value(self, 'azimuth_spin', settings.get('azimuth'))
            set_safe_widget_value(self, 'dip_spin', settings.get('dip'))
            
            # Grid origin
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            
            # Grid spacing
            set_safe_widget_value(self, 'grid_x_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'grid_y_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'grid_z_spin', settings.get('grid_z'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
            set_safe_widget_value(self, 'use_max_dist_check', settings.get('use_max_dist'))
            set_safe_widget_value(self, 'max_dist_spin', settings.get('max_dist'))
            set_safe_widget_value(self, 'auto_fit_grid_check', settings.get('auto_fit_grid'))
            
            # Toggle max_dist visibility
            if settings.get('use_max_dist') and hasattr(self, '_toggle_max_dist'):
                self._toggle_max_dist(2)  # Qt.CheckState.Checked.value
                
            logger.info("Restored kriging panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore kriging panel settings: {e}")
