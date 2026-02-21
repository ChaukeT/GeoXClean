"""
SGSIM Simulation & Uncertainty Analysis Panel
==============================================

UI for Sequential Gaussian Simulation with integrated post-processing,
uncertainty analysis, and visualization.
Refactored for modern UX/UI.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
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
    import matplotlib
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvasQTAgg = None
    Figure = None

from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QLabel,
    QMessageBox, QTabWidget, QWidget, QTextEdit,
    QFileDialog, QCheckBox, QLineEdit, QSplitter, QScrollArea, QFrame, QSizePolicy,
    QProgressBar, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
from datetime import datetime

from ..utils.coordinate_utils import ensure_xyz_columns
from ..utils.variable_utils import get_grade_columns, populate_variable_combo, validate_variable
from .base_analysis_panel import BaseAnalysisPanel, log_registry_data_status
from .modern_styles import get_theme_colors, get_analysis_panel_stylesheet

logger = logging.getLogger(__name__)


def get_sgsim_panel_stylesheet() -> str:
    """Enhanced stylesheet for SGSIM panel with high contrast fixes."""
    colors = get_theme_colors()
    return f"""
        /* Ensure all text is bright white/light grey */
        QLabel {{
            color: #FFFFFF;
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


class SGSIMPanel(BaseAnalysisPanel):
    """
    SGSIM Simulation & Uncertainty Analysis Panel.
    """
    # PanelManager metadata
    PANEL_ID = "SGSIMPanel"
    PANEL_NAME = "SGSIM Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT





    task_name = "sgsim"
    request_visualization = pyqtSignal(object, str)
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, parent=None):
        # State - initialize BEFORE super().__init__
        self.drillhole_data = None
        self.variable = None
        self.sgsim_results = None
        self.results_ready = False
        self.transformation_metadata = {}
        self.variogram_results = None
        self.selected_vcol = None
        # Optional: grid specification inferred from an existing block model
        # so SGSIM grid can align perfectly with the block model builder output.
        self.block_grid_spec = None
        # Main window reference for renderer access (set by MainWindow after creation)
        self.main_window = None
        
        super().__init__(parent=parent, panel_id="sgsim")
        self.setWindowTitle("SGSIM Simulation")
        self.resize(1200, 800)
        
        # Build UI (required when using _build_ui pattern)
        self._build_ui()

        self._init_registry()

        # Connect progress signal to update method
        self.progress_updated.connect(self._update_progress)
    
    def _build_ui(self):
        """Build custom split-pane UI. Called by base class."""
        self._setup_ui()
        
    def _init_registry(self):
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return
            
            # FIX: Check if signals are available before connecting
            dh_signal = self.registry.drillholeDataLoaded
            if dh_signal is not None:
                dh_signal.connect(self._on_data_loaded)
                logger.debug("SGSimPanel: Connected to drillholeDataLoaded signal")
            
            vario_signal = self.registry.variogramResultsLoaded
            if vario_signal is not None:
                vario_signal.connect(self._on_vario_loaded)
                logger.debug("SGSimPanel: Connected to variogramResultsLoaded signal")
            
            transform_signal = getattr(self.registry, 'transformationMetadataLoaded', None)
            if transform_signal is not None:
                transform_signal.connect(self._on_transformation_loaded)
                logger.debug("SGSimPanel: Connected to transformationMetadataLoaded signal")
            # Keep SGSIM results alive across panel close/reopen so the user
            # does not need to re‑run simulations just to visualize/export.
            if hasattr(self.registry, 'sgsimResultsLoaded'):
                try:
                    self.registry.sgsimResultsLoaded.connect(self._on_sgsim_loaded)
                except Exception:
                    # Older registry versions may not expose this signal
                    pass
            # Listen for block models generated by Block Model Builder or other panels
            if hasattr(self.registry, 'blockModelGenerated'):
                try:
                    self.registry.blockModelGenerated.connect(self._on_block_model_loaded)
                except Exception:
                    # Older registry versions may not expose this signal
                    pass
            
            # Source-toggle panels must load full drillhole payload for proper source switching.
            d = self.registry.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)
            v = self.registry.get_variogram_results()
            if v is not None:
                self._on_vario_loaded(v)
            # Retrieve transformation metadata if available
            t = self.registry.get_transformation_metadata() if hasattr(self.registry, 'get_transformation_metadata') else None
            if t:
                self.transformation_metadata = t
            # Restore last SGSIM results if they exist
            try:
                existing_sgsim = self.registry.get_sgsim_results()
            except Exception:
                existing_sgsim = None
            if existing_sgsim:
                self._restore_sgsim_results(existing_sgsim, source="registry")
            
            # If a block model already exists, try to infer its grid spec
            try:
                bm = self.registry.get_block_model()
                if bm is not None:
                    self._on_block_model_loaded(bm)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"DataRegistry connection failed: {e}", exc_info=True)
            self.registry = None
    
    def _on_transformation_loaded(self, metadata):
        """Handle transformation metadata loaded from registry."""
        self.transformation_metadata = metadata
    
    def _on_block_model_loaded(self, block_model):
        """
        Capture existing block model grid so SGSIM grid can be aligned.
        
        Supports both BlockModel objects and DataFrame-based block models
        registered by the Block Model Builder.
        """
        try:
            import pandas as pd  # Local import to avoid circulars in some test envs
            import numpy as np
        except Exception:
            return
        
        try:
            # Obtain a DataFrame view of block centers
            if hasattr(block_model, "to_dataframe"):
                df = block_model.to_dataframe()
            elif isinstance(block_model, pd.DataFrame):
                df = block_model.copy()
            else:
                return
            
            if df is None or df.empty:
                return
            
            # Normalise coordinate columns to X/Y/Z if possible
            df = ensure_xyz_columns(df)
            if not all(col in df.columns for col in ("X", "Y", "Z")):
                return
            
            x_coords = np.unique(df["X"].values.astype(float))
            y_coords = np.unique(df["Y"].values.astype(float))
            z_coords = np.unique(df["Z"].values.astype(float))
            
            if len(x_coords) == 0 or len(y_coords) == 0 or len(z_coords) == 0:
                return
            
            nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
            
            # Derive increments and origin from unique coordinates
            xinc = float(np.mean(np.diff(np.sort(x_coords)))) if nx > 1 else 1.0
            yinc = float(np.mean(np.diff(np.sort(y_coords)))) if ny > 1 else 1.0
            zinc = float(np.mean(np.diff(np.sort(z_coords)))) if nz > 1 else 1.0
            
            xmin = float(np.min(x_coords) - xinc / 2.0)
            ymin = float(np.min(y_coords) - yinc / 2.0)
            zmin = float(np.min(z_coords) - zinc / 2.0)
            
            self.block_grid_spec = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "xmin": xmin,
                "ymin": ymin,
                "zmin": zmin,
                "xinc": xinc,
                "yinc": yinc,
                "zinc": zinc,
            }
            
            logger.info(
                "SGSIMPanel: Detected existing block model grid "
                f"(nx={nx}, ny={ny}, nz={nz}, origin=({xmin:.2f},{ymin:.2f},{zmin:.2f}), "
                f"inc=({xinc:.2f},{yinc:.2f},{zinc:.2f}))"
            )
        except Exception as e:
            logger.warning(f"SGSIMPanel: Failed to infer grid from block model: {e}")
    
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

        # --- LEFT: CONFIGURATION ---
        left = QWidget()
        left.setStyleSheet(get_sgsim_panel_stylesheet())
        l_lay = QVBoxLayout(left)
        l_lay.setContentsMargins(10, 10, 10, 10)

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
        self.refresh_btn.setToolTip("Refresh data from registry (get latest composites/variogram)")
        self.refresh_btn.clicked.connect(self._manual_refresh)

        h_header.addWidget(header_title)
        h_header.addStretch()
        h_header.addWidget(self.refresh_btn)
        status_layout.addLayout(h_header)

        # Data source radio buttons (moved from data source group)
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

        # Data source status label (moved from data source group)
        self.data_source_status_label = QLabel("Initializing data...")
        self.data_source_status_label.setWordWrap(True)
        self.data_source_status_label.setStyleSheet("font-size: 9pt; color: #aaa; margin-top: 8px;")
        status_layout.addWidget(self.data_source_status_label)

        l_lay.addWidget(status_card)

        # 2. CONFIGURATION SCROLL AREA
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        cont = QWidget()
        s_lay = QVBoxLayout(cont)
        s_lay.setSpacing(15)

        self._create_data_source_group(s_lay)
        self._create_sim_settings(s_lay)
        self._create_grid_group(s_lay)
        self._create_variogram_group(s_lay)
        self._create_search_group(s_lay)
        self._create_cutoff_group(s_lay)

        s_lay.addStretch()
        scroll.setWidget(cont)
        l_lay.addWidget(scroll)

        # --- RIGHT: RESULTS & ANALYSIS ---
        right = QWidget()
        r_lay = QVBoxLayout(right)
        r_lay.setContentsMargins(10, 10, 10, 10)
        
        # Progress bar with percentage
        progress_group = QGroupBox("Progress")
        colors = get_theme_colors()
        progress_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v/%m")
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {colors.BORDER};
                border-radius: 4px;
                background-color: {colors.CARD_BG};
                text-align: center;
                color: {colors.TEXT_PRIMARY};
                height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: #4CAF50;
                border-radius: 3px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 10pt;")
        progress_layout.addWidget(self.progress_label)
        
        r_lay.addWidget(progress_group)
        
        # Event Log (like drillhole panel)
        r_lay.addWidget(QLabel("<b>Event Log</b>"))
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        colors = get_theme_colors()
        self.results_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas; font-size: 9pt;")
        self.results_text.setMaximumHeight(200)
        r_lay.addWidget(self.results_text)
        
        # Analysis Tools Tabs
        self.tabs = QTabWidget()
        self._create_viz_tab()
        self._create_uncert_tab()
        r_lay.addWidget(self.tabs, stretch=1)
        
        # Actions
        act_lay = QHBoxLayout()
        self.run_btn = QPushButton("RUN SGSIM")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 12px;")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results_menu)
        self.export_btn.setEnabled(False)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        
        act_lay.addWidget(self.run_btn, stretch=2)
        act_lay.addWidget(self.export_btn, stretch=1)
        act_lay.addWidget(self.clear_btn)
        r_lay.addLayout(act_lay)
        
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        layout.addWidget(splitter)

    # --- Configuration Groups ---

    def _create_data_source_group(self, layout):
        """Create data source selection group - now only contains banner.
        Data source selector moved to Data Status Card above."""
        g = QGroupBox("0. Data Source")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #81c784; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QFormLayout(g)
        l.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # New data notification banner (hidden by default) - Enhanced styling
        self._new_data_banner = QLabel("🔔 New composite data is available!")
        self._new_data_banner.setObjectName("NewDataBanner")
        self._new_data_banner.setWordWrap(True)
        self._new_data_banner.setVisible(False)
        l.addRow("", self._new_data_banner)

        layout.addWidget(g)

    def _on_data_source_changed(self, button):
        """Handle data source selection change."""
        if not hasattr(self, 'registry') or not self.registry:
            return

        # Reload full drillhole payload so panel can apply selected source correctly.
        data = self.registry.get_drillhole_data()
        if data is not None:
            self._on_data_loaded(data)

    def _create_sim_settings(self, layout):
        g = QGroupBox("1. Simulation Settings")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #4fc3f7; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QFormLayout(g)
        l.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        # Property/Variable selection
        self.variable_combo = QComboBox()
        self.variable_combo.setToolTip(
            "Select the grade variable to simulate using SGSIM.\n"
            "Sequential Gaussian Simulation generates multiple equiprobable realizations\n"
            "for uncertainty quantification and resource confidence intervals."
        )
        self.variable_combo.currentTextChanged.connect(self._on_variable_changed)
        l.addRow("Property/Variable:", self.variable_combo)

        self.nreal_spin = QSpinBox()
        self.nreal_spin.setRange(1, 1000)
        self.nreal_spin.setValue(50)
        self.nreal_spin.setToolTip(
            "Number of realizations to generate.\n"
            "Typical: 50-100 for resource modeling, 100-500 for risk analysis.\n"
            "More realizations = better uncertainty estimates but longer runtime."
        )
        
        # AUDIT FIX (W-001): Seed is MANDATORY for JORC/SAMREC reproducibility
        # Auto-generate a default seed based on current time to ensure reproducibility
        import time
        default_seed = int(time.time()) % 100000
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)  # Minimum 1, no more "Random" option
        self.seed_spin.setValue(default_seed)
        self.seed_spin.setToolTip(
            "Random seed for reproducibility (REQUIRED).\n"
            "Same seed = same results. Record this for JORC/SAMREC compliance."
        )
        
        # Add "Generate New Seed" button
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(self.seed_spin)
        self.gen_seed_btn = QPushButton("🎲")
        self.gen_seed_btn.setFixedWidth(30)
        self.gen_seed_btn.setToolTip("Generate new random seed")
        self.gen_seed_btn.clicked.connect(self._generate_new_seed)
        seed_layout.addWidget(self.gen_seed_btn)
        seed_widget = QWidget()
        seed_widget.setLayout(seed_layout)
        
        l.addRow("Realizations:", self.nreal_spin)
        l.addRow("Seed (required):", seed_widget)
        layout.addWidget(g)

    def _create_grid_group(self, layout):
        g = QGroupBox("2. Grid & Block Size")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffb74d; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)

        # Grid Origin (xmin, ymin, zmin)
        origin_label = QLabel("Grid Origin (corner of first block):")
        origin_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 9pt;")
        l.addWidget(origin_label)
        
        h0 = QHBoxLayout()
        self.xmin_spin = QDoubleSpinBox()
        self.xmin_spin.setRange(-1e9, 1e9)
        self.xmin_spin.setDecimals(1)
        self.xmin_spin.setValue(0)
        self.xmin_spin.setToolTip("X origin (min X coordinate of grid)")
        self.ymin_spin = QDoubleSpinBox()
        self.ymin_spin.setRange(-1e9, 1e9)
        self.ymin_spin.setDecimals(1)
        self.ymin_spin.setValue(0)
        self.ymin_spin.setToolTip("Y origin (min Y coordinate of grid)")
        self.zmin_spin = QDoubleSpinBox()
        self.zmin_spin.setRange(-1e9, 1e9)
        self.zmin_spin.setDecimals(1)
        self.zmin_spin.setValue(0)
        self.zmin_spin.setToolTip("Z origin (min Z/elevation coordinate of grid)")
        h0.addWidget(QLabel("X₀:"))
        h0.addWidget(self.xmin_spin)
        h0.addWidget(QLabel("Y₀:"))
        h0.addWidget(self.ymin_spin)
        h0.addWidget(QLabel("Z₀:"))
        h0.addWidget(self.zmin_spin)
        l.addLayout(h0)
        
        # Number of blocks (NX, NY, NZ)
        blocks_label = QLabel("Number of Blocks:")
        blocks_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(blocks_label)
        
        h1 = QHBoxLayout()
        self.nx = QSpinBox()
        self.nx.setRange(1, 1000)
        self.nx.setValue(50)
        self.ny = QSpinBox()
        self.ny.setRange(1, 1000)
        self.ny.setValue(50)
        self.nz = QSpinBox()
        self.nz.setRange(1, 1000)
        self.nz.setValue(20)
        h1.addWidget(QLabel("NX:"))
        h1.addWidget(self.nx)
        h1.addWidget(QLabel("NY:"))
        h1.addWidget(self.ny)
        h1.addWidget(QLabel("NZ:"))
        h1.addWidget(self.nz)
        l.addLayout(h1)
        
        # Block size (spacing)
        size_label = QLabel("Block Size (meters):")
        size_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        l.addWidget(size_label)
        
        h2 = QHBoxLayout()
        self.dx = QDoubleSpinBox()
        self.dx.setRange(0.1, 1000)
        self.dx.setValue(10)
        self.dx.setToolTip("Block size in X (meters) for SGSIM grid and visualization")
        self.dy = QDoubleSpinBox()
        self.dy.setRange(0.1, 1000)
        self.dy.setValue(10)
        self.dy.setToolTip("Block size in Y (meters) for SGSIM grid and visualization")
        self.dz = QDoubleSpinBox()
        self.dz.setRange(0.1, 1000)
        self.dz.setValue(5)
        self.dz.setToolTip("Block size in Z (meters) for SGSIM grid and visualization")
        h2.addWidget(QLabel("DX:"))
        h2.addWidget(self.dx)
        h2.addWidget(QLabel("DY:"))
        h2.addWidget(self.dy)
        h2.addWidget(QLabel("DZ:"))
        h2.addWidget(self.dz)
        l.addLayout(h2)
        
        auto_btn = QPushButton("Auto-Detect from Drillholes")
        auto_btn.setToolTip("Calculate grid origin and size to cover all drillhole data with padding")
        auto_btn.clicked.connect(self._auto_detect_grid)
        l.addWidget(auto_btn)
        
        # Auto-fit checkbox (enabled by default)
        self.auto_fit_grid_check = QCheckBox("Auto-fit grid when data loads")
        self.auto_fit_grid_check.setChecked(True)
        self.auto_fit_grid_check.setToolTip("Automatically restrict grid to drillhole extent when new data is loaded.\nPrevents simulation outside the data coverage area.")
        l.addWidget(self.auto_fit_grid_check)

        # Convenience preset for common cube size requested by users (e.g. 10×10×10 m)
        preset_btn = QPushButton("Set Block Size to 10 × 10 × 10 m")
        preset_btn.setToolTip("Quickly set DX, DY, DZ to 10 m for cubic blocks")
        preset_btn.clicked.connect(lambda: (self.dx.setValue(10.0), self.dy.setValue(10.0), self.dz.setValue(10.0)))
        l.addWidget(preset_btn)

        # Optional reblocking for visualization only (post‑SGSIM block mean)
        # NOTE: QGroupBox/QFormLayout/QCheckBox are already imported at module level.
        # Avoid re-importing them here, which would create a local QGroupBox name
        # and break the earlier use of QGroupBox in this function (UnboundLocalError).
        viz_group = QGroupBox("Visualization Reblock (Optional)")
        colors = get_theme_colors()
        viz_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ffe082; border: 1px dotted {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        vf = QFormLayout(viz_group)
        vf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        vf.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.viz_reblock_check = QCheckBox("Use coarser block size for visualization (block mean)")
        self.viz_reblock_check.setToolTip(
            "If checked, summary grids (mean, P10/P50/P90, probability) will be reblocked\n"
            "to a coarser block size for visualization only. Simulation is still run at the\n"
            "original DX/DY/DZ for accuracy."
        )
        vf.addRow(self.viz_reblock_check)

        # Visualization block sizes – default to simulation DX/DY/DZ
        self.viz_dx = QDoubleSpinBox()
        self.viz_dx.setRange(0.1, 1000)
        self.viz_dx.setValue(self.dx.value())
        self.viz_dx.setToolTip("Visualization block size in X (must be a multiple of DX to reblock cleanly)")

        self.viz_dy = QDoubleSpinBox()
        self.viz_dy.setRange(0.1, 1000)
        self.viz_dy.setValue(self.dy.value())
        self.viz_dy.setToolTip("Visualization block size in Y (must be a multiple of DY to reblock cleanly)")

        self.viz_dz = QDoubleSpinBox()
        self.viz_dz.setRange(0.1, 1000)
        self.viz_dz.setValue(self.dz.value())
        self.viz_dz.setToolTip("Visualization block size in Z (must be a multiple of DZ to reblock cleanly)")

        vf.addRow("Viz DX (m):", self.viz_dx)
        vf.addRow("Viz DY (m):", self.viz_dy)
        vf.addRow("Viz DZ (m):", self.viz_dz)

        # Enable / disable viz block size controls with checkbox
        def _toggle_viz_controls(checked: bool):
            self.viz_dx.setEnabled(checked)
            self.viz_dy.setEnabled(checked)
            self.viz_dz.setEnabled(checked)

        self.viz_reblock_check.toggled.connect(_toggle_viz_controls)
        _toggle_viz_controls(False)

        # Quick preset for 10×10×10 visualization blocks
        viz_preset_btn = QPushButton("Set Viz Block Size to 10 × 10 × 10 m")
        viz_preset_btn.setToolTip("Quickly set visualization DX, DY, DZ to 10 m; enable reblock if needed")

        def _set_viz_10():
            self.viz_reblock_check.setChecked(True)
            self.viz_dx.setValue(10.0)
            self.viz_dy.setValue(10.0)
            self.viz_dz.setValue(10.0)

        viz_preset_btn.clicked.connect(_set_viz_10)
        vf.addRow(viz_preset_btn)

        l.addWidget(viz_group)
        
        layout.addWidget(g)

    def _create_variogram_group(self, layout):
        g = QGroupBox("3. Variogram Model")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #ba68c8; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)
        
        # Load button
        load_btn = QPushButton("Load from Variogram Panel")
        load_btn.clicked.connect(self._load_from_variogram)
        l.addWidget(load_btn)
        
        # Type
        self.vario_type = QComboBox()
        self.vario_type.addItems(["Spherical", "Exponential", "Gaussian"])
        l.addWidget(self.vario_type)
        
        # Ranges
        h_range = QHBoxLayout()
        self.rmaj = QDoubleSpinBox()
        self.rmaj.setRange(1, 10000)
        self.rmaj.setValue(100)
        self.rmin = QDoubleSpinBox()
        self.rmin.setRange(1, 10000)
        self.rmin.setValue(50)
        self.rver = QDoubleSpinBox()
        self.rver.setRange(1, 10000)
        self.rver.setValue(25)
        
        r1 = QVBoxLayout()
        r1.addWidget(QLabel("Major:"))
        r1.addWidget(self.rmaj)
        r2 = QVBoxLayout()
        r2.addWidget(QLabel("Minor:"))
        r2.addWidget(self.rmin)
        r3 = QVBoxLayout()
        r3.addWidget(QLabel("Vert:"))
        r3.addWidget(self.rver)
        h_range.addLayout(r1)
        h_range.addLayout(r2)
        h_range.addLayout(r3)
        l.addLayout(h_range)
        
        # Angles & Sill
        h_param = QHBoxLayout()
        self.azim = QDoubleSpinBox()
        self.azim.setRange(0, 360)
        self.dip = QDoubleSpinBox()
        self.dip.setRange(-90, 90)
        self.nug = QDoubleSpinBox()
        self.nug.setRange(0, 100)
        self.sill = QDoubleSpinBox()
        self.sill.setValue(1.0)
        
        p1 = QVBoxLayout()
        p1.addWidget(QLabel("Azim:"))
        p1.addWidget(self.azim)
        p2 = QVBoxLayout()
        p2.addWidget(QLabel("Dip:"))
        p2.addWidget(self.dip)
        p3 = QVBoxLayout()
        p3.addWidget(QLabel("Nug:"))
        p3.addWidget(self.nug)
        p4 = QVBoxLayout()
        p4.addWidget(QLabel("Sill:"))
        p4.addWidget(self.sill)
        
        h_param.addLayout(p1)
        h_param.addLayout(p2)
        h_param.addLayout(p3)
        h_param.addLayout(p4)
        l.addLayout(h_param)
        
        layout.addWidget(g)

    def _create_search_group(self, layout):
        g = QGroupBox("4. Search")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #90a4ae; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QHBoxLayout(g)
        
        self.min_n = QSpinBox()
        self.min_n.setValue(8)  # Increased from 4 for better numerical stability
        self.max_n = QSpinBox()
        self.max_n.setValue(16)  # Increased from 12 for better conditioning
        self.rad = QDoubleSpinBox()
        self.rad.setRange(1, 10000)
        self.rad.setValue(200)
        
        l.addWidget(QLabel("Min:"))
        l.addWidget(self.min_n)
        l.addWidget(QLabel("Max:"))
        l.addWidget(self.max_n)
        l.addWidget(QLabel("Rad:"))
        l.addWidget(self.rad)
        layout.addWidget(g)

    def _create_cutoff_group(self, layout):
        g = QGroupBox("5. Uncertainty Cutoffs")
        colors = get_theme_colors()
        g.setStyleSheet(f"QGroupBox {{ font-weight: bold; color: #81c784; border: 1px solid {colors.BORDER}; margin-top: 6px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        l = QVBoxLayout(g)

        # Cutoff input row with auto-suggest button
        cutoff_row = QHBoxLayout()
        self.cutoff_edit = QLineEdit("")
        self.cutoff_edit.setPlaceholderText("Click 'Auto-Suggest' or enter comma separated values")
        cutoff_row.addWidget(self.cutoff_edit, stretch=3)

        self.auto_suggest_btn = QPushButton("Auto-Suggest")
        self.auto_suggest_btn.setToolTip("Suggest cutoff values based on data percentiles (P25, P50, P75, P90)")
        self.auto_suggest_btn.setStyleSheet("background-color: #388e3c; color: white;")
        self.auto_suggest_btn.clicked.connect(self._auto_suggest_cutoffs)
        cutoff_row.addWidget(self.auto_suggest_btn, stretch=1)

        l.addLayout(cutoff_row)
        layout.addWidget(g)

    def _auto_suggest_cutoffs(self):
        """Auto-suggest cutoff values based on data percentiles.

        Uses P25, P50, P75, P90 percentiles to provide meaningful cutoffs
        that span the grade distribution. Works for any commodity.
        """
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "Load data first to auto-suggest cutoffs.")
            return

        # Get selected variable
        variable = self.variable_combo.currentText() if hasattr(self, 'variable_combo') else None
        if not variable or variable not in self.drillhole_data.columns:
            QMessageBox.warning(self, "No Variable", "Select a property/variable first.")
            return

        try:
            values = self.drillhole_data[variable].dropna()
            if len(values) == 0:
                QMessageBox.warning(self, "No Data", f"No valid data for '{variable}'.")
                return

            # Calculate percentiles
            p25 = values.quantile(0.25)
            p50 = values.quantile(0.50)
            p75 = values.quantile(0.75)
            p90 = values.quantile(0.90)

            # Determine decimal places based on data magnitude
            grade_max = values.max()
            if grade_max > 50:
                decimals = 1  # High values (e.g., Fe% 30-65)
            elif grade_max > 10:
                decimals = 1  # Medium values
            elif grade_max > 1:
                decimals = 2  # Lower values (e.g., Cu% 0.1-2)
            else:
                decimals = 3  # Very low values (e.g., Au g/t)

            # Round percentiles
            cutoffs = [round(p25, decimals), round(p50, decimals), round(p75, decimals), round(p90, decimals)]
            # Remove duplicates while preserving order
            unique_cutoffs = []
            for c in cutoffs:
                if c not in unique_cutoffs:
                    unique_cutoffs.append(c)

            # Format as comma-separated string
            cutoff_str = ", ".join(str(c) for c in unique_cutoffs)
            self.cutoff_edit.setText(cutoff_str)

            # Log suggestion
            logger.info(f"SGSIM: Auto-suggested cutoffs for '{variable}': {cutoff_str} "
                       f"(P25={p25:.3f}, P50={p50:.3f}, P75={p75:.3f}, P90={p90:.3f})")
            self._log_event(f"Auto-suggested cutoffs: {cutoff_str}", "info")

        except Exception as e:
            logger.warning(f"SGSIM: Could not auto-suggest cutoffs: {e}")
            QMessageBox.warning(self, "Error", f"Could not calculate cutoffs: {e}")

    # --- Right Panel Tabs ---

    def _create_viz_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        
        self.back_transform_btn = QPushButton("Back-Transform (Gaussian → Grade)")
        self.back_transform_btn.setStyleSheet("background-color: #7b1fa2; color: white;")
        self.back_transform_btn.clicked.connect(self._back_transform_results)
        self.back_transform_btn.setEnabled(False)
        l.addWidget(self.back_transform_btn)

        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        grid.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        h1 = QHBoxLayout()
        self.viz_mean = QPushButton("Mean")
        self.viz_mean.clicked.connect(lambda: self._visualize_summary("mean"))
        self.viz_std = QPushButton("Std Dev")
        self.viz_std.clicked.connect(lambda: self._visualize_summary("std"))
        h1.addWidget(self.viz_mean)
        h1.addWidget(self.viz_std)
        grid.addRow("Summary:", h1)
        
        h2 = QHBoxLayout()
        self.viz_p10 = QPushButton("P10")
        self.viz_p10.clicked.connect(lambda: self._visualize_summary("p10"))
        self.viz_p50 = QPushButton("P50")
        self.viz_p50.clicked.connect(lambda: self._visualize_summary("p50"))
        self.viz_p90 = QPushButton("P90")
        self.viz_p90.clicked.connect(lambda: self._visualize_summary("p90"))
        h2.addWidget(self.viz_p10)
        h2.addWidget(self.viz_p50)
        h2.addWidget(self.viz_p90)
        grid.addRow("Percentiles:", h2)
        
        l.addLayout(grid)
        
        # Disable initially
        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90]:
            b.setEnabled(False)
            
        l.addStretch()
        self.tabs.addTab(tab, "Visualization")

    def _create_uncert_tab(self):
        tab = QWidget()
        l = QVBoxLayout(tab)
        
        self.uncert_text = QTextEdit()
        self.uncert_text.setReadOnly(True)
        colors = get_theme_colors()
        self.uncert_text.setStyleSheet(f"background-color: {colors.CARD_BG}; color: {colors.TEXT_PRIMARY}; font-family: Consolas;")
        l.addWidget(self.uncert_text)
        
        l.addWidget(QLabel("Probability Maps (P > Cutoff):"))
        self.prob_layout = QHBoxLayout()  # Dynamically populated
        l.addLayout(self.prob_layout)
        
        l.addStretch()
        self.tabs.addTab(tab, "Uncertainty")

    # --- Logic ---

    def _on_data_loaded(self, data):
        """Handle drillhole data loaded from registry.

        Shows notification banner if panel is visible, otherwise marks for refresh.
        """
        # Store data for later use
        self._pending_drillhole_data = data

        # If panel is visible, show notification banner (user decides when to refresh)
        if self.isVisible():
            if hasattr(self, '_new_data_banner'):
                self._new_data_banner.setVisible(True)
                self._pending_data_update = True
                logger.info("SGSIMPanel: New data available, notification shown")
            return

        # Panel not visible - process immediately
        self._apply_data_update(data)

    def _apply_data_update(self, data):
        """Apply the data update to UI controls."""
        # Log diagnostic info about registry contents
        data_source = log_registry_data_status("SGSIM", data)

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
        
        if df is not None and not df.empty:
            self._process_drillhole_data(df)
        else:
            logger.warning("SGSIM: Received empty or None drillhole data")

        # Hide notification banner after update
        if hasattr(self, '_new_data_banner'):
            self._new_data_banner.setVisible(False)
        self._pending_data_update = False

    def _manual_refresh(self):
        """Manual refresh - reload data from registry."""
        try:
            registry = self.get_registry()
            if registry:
                # Get latest drillhole data
                data = registry.get_drillhole_data()
                if data:
                    self._apply_data_update(data)

                # Also try to get latest variogram results
                vario = registry.get_data("variogram_results")
                if vario:
                    self.set_variogram_results(vario)

                # Get latest transformation metadata
                trans = registry.get_data("transformation_metadata")
                if trans:
                    self._on_transformation_loaded(trans)

                # Hide notification banner
                if hasattr(self, '_new_data_banner'):
                    self._new_data_banner.setVisible(False)
                self._pending_data_update = False

                # Show feedback
                if hasattr(self, 'log_text') and self.log_text is not None:
                    self._log_event("Data refreshed from registry", "info")
                logger.info("SGSIMPanel: Manual refresh completed")
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}", exc_info=True)
            if hasattr(self, 'log_text') and self.log_text is not None:
                self._log_event(f"Refresh failed: {e}", "error")

    def showEvent(self, event):
        """Auto-refresh when panel becomes visible."""
        super().showEvent(event)

        # If there's a pending update, apply it now
        if getattr(self, '_pending_data_update', False):
            if hasattr(self, '_pending_drillhole_data') and self._pending_drillhole_data is not None:
                self._apply_data_update(self._pending_drillhole_data)

    def set_drillhole_data(self, df, variable=None):
        """
        Legacy compatibility method - delegates to registry-based data loading.
        New code should use registry.drillholeDataLoaded signal.
        """
        # Delegate to internal processing method
        if df is not None:
            self._process_drillhole_data(df, variable)
    
    def _process_drillhole_data(self, df, variable=None):
        """Internal method to process drillhole data (called by both set_drillhole_data and _on_data_loaded)."""
        if df is not None and not df.empty:
            self.drillhole_data = ensure_xyz_columns(df)
            
            # Populate property combo box using centralized utility
            if self.drillhole_data is not None and hasattr(self, 'variable_combo'):
                valid_cols = get_grade_columns(self.drillhole_data)
                
                # Use centralized utility to populate combo
                selected = populate_variable_combo(self.variable_combo, self.drillhole_data, variable)
            
            # Check metadata for transformed vars - look for transformed column names
            if self.transformation_metadata:
                # First, check if any transformed columns exist in the data
                for col in valid_cols:
                    for k, v in self.transformation_metadata.items():
                        if isinstance(v, dict):
                            transformed_name = v.get('transformed_col_name')
                            original_name = v.get('original_col_name', k)
                            # If this column matches a transformed name, prefer it
                            if transformed_name == col:
                                variable = col
                                break
                            # If we're looking for a transformed column that doesn't exist,
                            # check if the original exists and we can use it
                            elif transformed_name == variable and original_name in valid_cols:
                                # The transformed column doesn't exist, but original does
                                # We'll use the original and transform it later if needed
                                logger.info(f"Transformed column '{transformed_name}' not found, but original '{original_name}' exists")
                                variable = original_name
                                break
                    if variable:
                        break
            
            # Set the selected variable (may have been updated by transformation logic)
            if variable and variable in valid_cols:
                self.variable_combo.setCurrentText(variable)
                self.variable = variable
            elif selected:
                self.variable = selected
            elif valid_cols:
                self.variable = valid_cols[0]
            else:
                self.variable = None
        
        # Only update UI if it's been built
        if hasattr(self, 'run_btn') and hasattr(self, 'results_text'):
            if self.variable:
                self.run_btn.setEnabled(True)
                self.results_text.clear()
                self._log_event("SGSIM Panel Initialized", "success")
                self._log_event(f"Data loaded: {len(self.drillhole_data)} samples", "info")
                self._log_event(f"Variable: {self.variable}", "info")
                self._log_event(f"Available columns: {len(valid_cols)}", "info")
                self._update_progress(0, "Ready to run")
                
                # Auto-detect grid from drillhole extent if checkbox is enabled
                if hasattr(self, 'auto_fit_grid_check') and self.auto_fit_grid_check.isChecked():
                    try:
                        self._auto_detect_grid()
                        self._log_event("Auto-fitted grid to drillhole extent", "success")
                        logger.info("SGSIM panel: Auto-fitted grid to drillhole extent")
                    except Exception as e:
                        logger.warning(f"SGSIM panel: Auto-fit grid failed: {e}")
            else:
                self.run_btn.setEnabled(False)
                self._log_event("No valid properties found in data", "warning")
                logger.debug("SGSIM panel: No valid properties found")
    
    def _on_variable_changed(self, text):
        """Handle variable selection change."""
        if text:
            self.variable = text
            if hasattr(self, 'results_text'):
                self.results_text.setText(f"Variable changed to: {self.variable}")
            logger.info(f"SGSIM panel: Variable changed to {self.variable}")

    def _generate_new_seed(self):
        """Generate a new random seed for simulation reproducibility."""
        import time
        new_seed = int(time.time() * 1000) % 999999
        self.seed_spin.setValue(max(1, new_seed))  # Ensure minimum of 1
        self._log_event(f"Generated new seed: {new_seed}", "info")

    def _on_vario_loaded(self, res):
        self.set_variogram_results(res)

    def set_variogram_results(self, res):
        self.variogram_results = res
        if not res:
            return
        
        # Only update UI if it's been built
        if not (hasattr(self, 'vario_type') and hasattr(self, 'rmaj') and 
                hasattr(self, 'sill') and hasattr(self, 'nug') and
                hasattr(self, 'rmin') and hasattr(self, 'rver') and
                hasattr(self, 'azim') and hasattr(self, 'dip')):
            logger.debug("SGSIM panel: UI not ready for variogram parameter loading")
            return
            
        # Load logic - extract fitted model parameters
        models = res.get('fitted_models', {}).get('omni', {})
        sill = 1.0  # Default for validation
        if models:
            m_type = list(models.keys())[0]
            p = models[m_type]

            self.vario_type.setCurrentText(m_type.capitalize())
            self.rmaj.setValue(p.get('range', 100))

            # ✅ FIX: Use 'total_sill' (nugget + partial_sill), not 'sill' (which is partial_sill)
            # The variogram fitting stores 'sill' as partial_sill, 'total_sill' as nugget + partial_sill
            # SGSIM expects total_sill for proper simulation
            nugget = p.get('nugget', 0.0)
            if 'total_sill' in p:
                sill = p.get('total_sill')
            else:
                # Fallback: if only 'sill' is available, it's partial sill, so add nugget
                partial_sill = p.get('sill', 1.0)
                sill = nugget + partial_sill

            self.nug.setValue(nugget)
            self.sill.setValue(sill)

            logger.info(f"SGSIM: Loaded variogram params - nugget={nugget:.3f}, sill={sill:.3f} (total), range={p.get('range', 100):.1f}")

            # Load directional ranges if available, otherwise use omni range with scaling
            omni_range = p.get('range', 100)

            # Try to get directional ranges from fitted_models
            fitted = res.get('fitted_models', {})
            major_range = omni_range
            minor_range = omni_range * 0.7  # Default: 70% of major
            vert_range = omni_range * 0.5   # Default: 50% of major

            # Check for directional models
            if 'major' in fitted and fitted['major']:
                major_model = list(fitted['major'].values())[0]
                major_range = major_model.get('range', major_range)

            if 'minor' in fitted and fitted['minor']:
                minor_model = list(fitted['minor'].values())[0]
                minor_range = minor_model.get('range', minor_range)

            if 'vertical' in fitted and fitted['vertical']:
                vert_model = list(fitted['vertical'].values())[0]
                vert_range = vert_model.get('range', vert_range)
            # NOTE: Do NOT fall back to downhole for vertical range!
            # Downhole is for nugget estimation, not vertical anisotropy.
            # If no vertical fit available, use scaled omni range (default)

            self.rmaj.setValue(major_range)
            self.rmin.setValue(minor_range)
            self.rver.setValue(vert_range)

            logger.info(f"SGSIM: Directional ranges - major={major_range:.1f}, minor={minor_range:.1f}, vert={vert_range:.1f}")

        self.azim.setValue(res.get('azimuth', 0))
        self.dip.setValue(res.get('dip', 0))

        # ⚠️ VALIDATION: Check if variogram parameters are appropriate for SGSIM (normal-score domain)
        # Normal-score data should have sill ≈ 1.0, while raw grade data has much higher variance
        if sill > 2.0 or sill < 0.5:  # Outside reasonable range for normal-score data
            warning_msg = (
                f"⚠️ Variogram Validation Warning:\n\n"
                f"Loaded variogram sill = {sill:.3f}\n\n"
                f"For SGSIM to work correctly, variogram parameters MUST be fitted on "
                f"normal-score transformed data (sill should be ≈ 1.0).\n\n"
                f"If these parameters came from raw grade data, SGSIM results will be "
                f"geostatistically incorrect and may show poor mean/variance reproduction.\n\n"
                f"Recommendation: Fit variogram on normal-score transformed data first, "
                f"then use those parameters for SGSIM."
            )
            self._log_event("WARNING: Variogram may not be fitted on normal-score data", "warning")
            QMessageBox.warning(self, "Variogram Domain Warning", warning_msg)

        # Check variable match
        v = res.get('variable')
        if v and v != self.variable:
            # Try to set it in the combo box if it exists
            if hasattr(self, 'variable_combo') and self.drillhole_data is not None:
                if v in self.drillhole_data.columns:
                    self.variable_combo.setCurrentText(v)
                    self.variable = v
                    if hasattr(self, 'results_text'):
                        self.results_text.append(f"Switched variable to {v} based on variogram.")
                else:
                    # Check transformation metadata to see if this is a transformed column
                    original_col = None
                    if self.transformation_metadata:
                        for key, meta in self.transformation_metadata.items():
                            if isinstance(meta, dict):
                                transformed_name = meta.get('transformed_col_name')
                                original_name = meta.get('original_col_name', key)
                                if transformed_name == v and original_name in self.drillhole_data.columns:
                                    original_col = original_name
                                    break
                    
                    if original_col:
                        logger.warning(f"Variogram variable '{v}' not found, but original '{original_col}' exists. "
                                     f"Please ensure '{v}' is created via Grade Transformation panel before running SGSIM.")
                        if hasattr(self, 'results_text'):
                            self.results_text.append(f"⚠️ Warning: Variogram uses '{v}' but data only has '{original_col}'. "
                                                    f"Please transform '{original_col}' to '{v}' first.")
                        # Don't auto-switch, let user decide
                    else:
                        logger.warning(f"Variogram variable '{v}' not found in data columns. Available: {list(self.drillhole_data.columns)}")
                        if hasattr(self, 'results_text'):
                            self.results_text.append(f"⚠️ Warning: Variogram variable '{v}' not found in data.")
            else:
                self.variable = v
                if hasattr(self, 'results_text'):
                    self.results_text.append(f"Switched variable to {v} based on variogram.")

    def _auto_detect_grid(self):
        """
        Auto-detect grid parameters from drillhole coordinates.
        
        Uses the actual drillhole data extent to define a grid that:
        1. Covers all drillhole sample locations with padding
        2. Uses sensible default block sizes (10×10×5 m for typical mining)
        3. Aligns grid origin to round numbers for cleaner coordinates
        
        IMPORTANT: The grid MUST encompass all drillhole data points.
        - Grid starts at ZMIN (lowest drillhole point) and extends upward
        - This ensures the simulation covers the full vertical extent of the data
        
        CRITICAL FIX: Prefer actual rendered drillhole bounds over DataFrame bounds
        to ensure grid aligns with what's visually displayed.
        """
        if self.drillhole_data is None or self.drillhole_data.empty:
            QMessageBox.warning(self, "No Data", "No drillhole data loaded for grid detection.")
            return
        
        # Try to get actual rendered drillhole bounds first (most accurate)
        rendered_bounds = None
        renderer = self._find_renderer()  # Use helper method
        
        if renderer:
            logger.info(f"Auto-detect: Found renderer of type {type(renderer).__name__}")
            has_cache = hasattr(renderer, '_drillhole_polylines_cache')
            cache = getattr(renderer, '_drillhole_polylines_cache', None) if has_cache else None
            
            logger.info(f"Auto-detect: has_cache={has_cache}, cache_is_none={cache is None}")
            
            if cache and 'hole_polys' in cache:
                try:
                    hole_polys = cache['hole_polys']
                    logger.info(f"Auto-detect: Found {len(hole_polys)} hole polylines in cache")
                    
                    all_points = []
                    for hole_id, poly in hole_polys.items():
                        if hasattr(poly, 'points') and poly.n_points > 0:
                            all_points.append(poly.points)
                    
                    logger.info(f"Auto-detect: Collected points from {len(all_points)} polylines")
                    
                    if all_points:
                        stacked_points = np.vstack(all_points)
                        logger.info(f"Auto-detect: Total {len(stacked_points)} points, shape={stacked_points.shape}")
                        
                        rendered_bounds = {
                            'x_min': float(stacked_points[:, 0].min()),
                            'x_max': float(stacked_points[:, 0].max()),
                            'y_min': float(stacked_points[:, 1].min()),
                            'y_max': float(stacked_points[:, 1].max()),
                            'z_min': float(stacked_points[:, 2].min()),
                            'z_max': float(stacked_points[:, 2].max()),
                        }
                        logger.info(
                            f"Auto-detect: Using RENDERED drillhole bounds: "
                            f"X=[{rendered_bounds['x_min']:.2f}, {rendered_bounds['x_max']:.2f}], "
                            f"Y=[{rendered_bounds['y_min']:.2f}, {rendered_bounds['y_max']:.2f}], "
                            f"Z=[{rendered_bounds['z_min']:.2f}, {rendered_bounds['z_max']:.2f}]"
                        )
                    else:
                        logger.warning("Auto-detect: No valid polyline points found in cache")
                except Exception as e:
                    logger.warning(f"Could not extract rendered drillhole bounds: {e}", exc_info=True)
            else:
                if cache is None:
                    logger.warning("Auto-detect: Drillhole cache is None (drillholes not yet rendered?)")
                elif 'hole_polys' not in cache:
                    logger.warning(f"Auto-detect: Cache exists but 'hole_polys' key missing. Available keys: {list(cache.keys()) if cache else 'N/A'}")
        else:
            logger.warning("Auto-detect: Could not find renderer via any method")
        
        # Use rendered bounds if available, otherwise fall back to DataFrame
        if rendered_bounds:
            logger.info("Auto-detect: ✓ Successfully retrieved RENDERED drillhole bounds - grid will align with displayed drillholes")
            x_min = rendered_bounds['x_min']
            x_max = rendered_bounds['x_max']
            y_min = rendered_bounds['y_min']
            y_max = rendered_bounds['y_max']
            z_min = rendered_bounds['z_min']  # Deepest point (bottom of drillholes)
            z_max = rendered_bounds['z_max']  # Shallowest point (near surface)
            logger.info("Auto-detect: Using rendered drillhole bounds for grid detection")
        else:
            # Fall back to DataFrame bounds
            df = self.drillhole_data
            if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
                QMessageBox.warning(self, "Missing Coordinates", 
                                  "Drillhole data must have X, Y, Z columns for grid detection.")
                return
            
            # Get drillhole coordinate bounds (filter out NaN/inf values)
            valid_mask = df['X'].notna() & df['Y'].notna() & df['Z'].notna()
            valid_df = df[valid_mask]
            
            if valid_df.empty:
                QMessageBox.warning(self, "No Valid Data", "No valid coordinate data found in drillholes.")
                return
            
            x_min = float(valid_df['X'].min())
            x_max = float(valid_df['X'].max())
            y_min = float(valid_df['Y'].min())
            y_max = float(valid_df['Y'].max())
            z_min = float(valid_df['Z'].min())  # Deepest point (bottom of drillholes)
            z_max = float(valid_df['Z'].max())  # Shallowest point (near surface)
            logger.info(
                f"Auto-detect: Using DataFrame bounds: "
                f"X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]"
            )
        
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
        dx = self.dx.value() if self.dx.value() > 0.1 else 10.0
        dy = self.dy.value() if self.dy.value() > 0.1 else 10.0
        dz = self.dz.value() if self.dz.value() > 0.1 else 5.0
        
        # Add padding: 1 block on each side minimum, or 5% of range
        x_pad = max(dx, x_range * 0.05)
        y_pad = max(dy, y_range * 0.05)
        z_pad = max(dz, z_range * 0.05)
        
        # Calculate grid origin (snap to block size for cleaner coordinates)
        # Origin is at the corner of the first block (block centers are at origin + dx/2, etc.)
        # CRITICAL: zmin must be at or BELOW the lowest drillhole point
        xmin = np.floor((x_min - x_pad) / dx) * dx
        ymin = np.floor((y_min - y_pad) / dy) * dy
        zmin = np.floor((z_min - z_pad) / dz) * dz  # Start BELOW the deepest point
        
        # Calculate grid extent - must cover all data plus padding
        xmax = np.ceil((x_max + x_pad) / dx) * dx
        ymax = np.ceil((y_max + y_pad) / dy) * dy
        zmax = np.ceil((z_max + z_pad) / dz) * dz  # End ABOVE the highest point
        
        # Calculate number of blocks needed to cover the extent
        nx = int(np.round((xmax - xmin) / dx))
        ny = int(np.round((ymax - ymin) / dy))
        nz = int(np.round((zmax - zmin) / dz))
        
        # Ensure at least 1 block in each dimension
        nx = max(1, nx)
        ny = max(1, ny)
        nz = max(1, nz)
        
        # Verify the grid actually covers the data (sanity check)
        grid_z_top = zmin + nz * dz
        if grid_z_top < z_max:
            # Grid doesn't reach the top - add more blocks
            nz = int(np.ceil((z_max + z_pad - zmin) / dz))
        
        # Set the UI values (origin, block count, block size)
        self.xmin_spin.setValue(xmin)
        self.ymin_spin.setValue(ymin)
        self.zmin_spin.setValue(zmin)
        self.dx.setValue(dx)
        self.dy.setValue(dy)
        self.dz.setValue(dz)
        self.nx.setValue(nx)
        self.ny.setValue(ny)
        self.nz.setValue(nz)
        
        # Log and display results
        total_blocks = nx * ny * nz
        grid_x_max = xmin + nx * dx
        grid_y_max = ymin + ny * dy
        grid_z_max = zmin + nz * dz
        
        # Verify grid covers all drillhole data
        covers_x = (xmin <= x_min) and (grid_x_max >= x_max)
        covers_y = (ymin <= y_min) and (grid_y_max >= y_max)
        covers_z = (zmin <= z_min) and (grid_z_max >= z_max)
        
        logger.info(
            f"Auto-detected grid from drillholes: {nx}×{ny}×{nz} = {total_blocks:,} blocks, "
            f"origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f}), "
            f"extent=({grid_x_max:.1f}, {grid_y_max:.1f}, {grid_z_max:.1f}), "
            f"spacing=({dx:.1f}, {dy:.1f}, {dz:.1f})m"
        )
        logger.info(
            f"  Drillhole data extent: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]"
        )
        logger.info(
            f"  Grid coverage check: X={covers_x}, Y={covers_y}, Z={covers_z} (all should be True)"
        )
        
        if hasattr(self, 'results_text'):
            self.results_text.append(
                f"<b>Grid Auto-Detection Results:</b><br>"
                f"  Drillhole extent: X=[{x_min:.1f}, {x_max:.1f}], "
                f"Y=[{y_min:.1f}, {y_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]<br>"
                f"  Grid origin (corner): ({xmin:.1f}, {ymin:.1f}, {zmin:.1f})<br>"
                f"  Grid Z-extent: [{zmin:.1f}, {grid_z_top:.1f}] (covers drillhole Z range)<br>"
                f"  Grid size: {nx} × {ny} × {nz} = <b>{total_blocks:,}</b> blocks<br>"
                f"  Block size: {dx:.1f} × {dy:.1f} × {dz:.1f} m"
            )
    
    def gather_parameters(self) -> Dict[str, Any]:
        if self.drillhole_data is None:
            raise ValueError("No data")
        
        # Get selected variable from combo box
        selected_variable = None
        if hasattr(self, 'variable_combo') and self.variable_combo.currentText():
            selected_variable = self.variable_combo.currentText()
        elif self.variable:
            selected_variable = self.variable
        else:
            raise ValueError("No variable/property selected. Please select a property from the dropdown.")
        
        # Verify the variable exists in the data
        if selected_variable not in self.drillhole_data.columns:
            # Check if it's a transformed column that should exist
            original_col = None
            if self.transformation_metadata:
                for key, meta in self.transformation_metadata.items():
                    if isinstance(meta, dict):
                        transformed_name = meta.get('transformed_col_name')
                        original_name = meta.get('original_col_name', key)
                        if transformed_name == selected_variable:
                            original_col = original_name
                            break
            
            if original_col and original_col in self.drillhole_data.columns:
                raise ValueError(
                    f"Selected variable '{selected_variable}' (transformed version of '{original_col}') not found in data. "
                    f"Please apply the transformation to '{original_col}' using the Grade Transformation panel first, "
                    f"or select '{original_col}' instead."
                )
            else:
                raise ValueError(
                    f"Selected variable '{selected_variable}' not found in data columns: {list(self.drillhole_data.columns)}"
                )
        
        # Cutoffs
        try:
            cuts = [float(x) for x in self.cutoff_edit.text().split(',') if x.strip()]
        except ValueError:
            cuts = []
        
        # Use origin values from the UI spinboxes (user can edit these)
        xmin = self.xmin_spin.value()
        ymin = self.ymin_spin.value()
        zmin = self.zmin_spin.value()
        
        return {
            "data_df": self.drillhole_data,
            "variable": selected_variable,
            "nreal": self.nreal_spin.value(),
            "seed": self.seed_spin.value(),  # AUDIT FIX (W-001): Seed is always required
            
            "nx": self.nx.value(),
            "ny": self.ny.value(),
            "nz": self.nz.value(),
            "xmin": xmin,
            "ymin": ymin,
            "zmin": zmin,
            "xinc": self.dx.value(),
            "yinc": self.dy.value(),
            "zinc": self.dz.value(),
            
            "variogram_type": self.vario_type.currentText().lower(),
            "range_major": self.rmaj.value(),
            "range_minor": self.rmin.value(),
            "range_vert": self.rver.value(),
            "azimuth": self.azim.value(),
            "dip": self.dip.value(),
            "nugget": self.nug.value(),
            "sill": self.sill.value(),
            
            "min_neighbors": self.min_n.value(),
            "max_neighbors": self.max_n.value(),
            "max_search_radius": self.rad.value(),
            "cutoffs": cuts,
            "transformation_metadata": self.transformation_metadata
        }

    def validate_inputs(self) -> bool:
        # Check if variable combo exists and has a selection
        if hasattr(self, 'variable_combo'):
            selected = self.variable_combo.currentText()
            if not selected:
                QMessageBox.warning(self, "Error", "Please select a property/variable to simulate from the dropdown.")
                return False
            # Verify it exists in data
            if self.drillhole_data is not None and selected not in self.drillhole_data.columns:
                QMessageBox.warning(self, "Error", f"Selected property '{selected}' not found in data.")
                return False
            self.variable = selected
        elif self.variable is None:
            QMessageBox.warning(self, "Error", "No variable selected. Please select a property from the dropdown.")
            return False
        
        if self.drillhole_data is None:
            QMessageBox.warning(self, "Error", "No drillhole data loaded.")
            return False
        
        # ✅ Validate variogram parameters to prevent NaN results
        nugget = self.nug.value()
        sill = self.sill.value()
        partial_sill = sill - nugget
        
        if partial_sill <= 0:
            reply = QMessageBox.warning(
                self, 
                "Invalid Variogram Parameters",
                f"Nugget ({nugget:.3f}) >= Sill ({sill:.3f}).\n\n"
                f"This will produce invalid simulation results (NaN values).\n\n"
                f"Please adjust the variogram parameters:\n"
                f"• Sill should be greater than Nugget\n"
                f"• Partial sill (Sill - Nugget) should be positive\n\n"
                f"Would you like to proceed anyway? (Not recommended)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return False
            else:
                self._log_event(f"⚠️ WARNING: Proceeding with nugget >= sill (may cause NaN)", "warning")
        elif partial_sill < 0.05:
            self._log_event(
                f"⚠️ Warning: Small partial sill ({partial_sill:.3f}). "
                f"Consider checking variogram parameters.", 
                "warning"
            )
        
        return True
    
    def show_progress(self, message: str) -> None:
        """Override to use built-in progress bar instead of modal dialog."""
        # Ensure progress bar is visible
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setVisible(True)
        if hasattr(self, 'progress_label') and self.progress_label:
            self.progress_label.setVisible(True)
        
        self._update_progress(0, message)
        self._log_event(f"Starting: {message}", "progress")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
    
    def hide_progress(self) -> None:
        """Override to use built-in progress bar."""
        self.run_btn.setEnabled(True)
        self.run_btn.setText("RUN SGSIM")
    
    def _check_data_lineage(self) -> bool:
        """
        HARD GATE: Verify data lineage before SGSIM simulation.

        SGSIM requires properly prepared data:
        1. QC-Validated (MUST pass or warn - HARD STOP on FAIL/NOT_RUN)
        2. Composited data (consistent sample support)
        3. Optionally declustered (for bias correction)
        4. Transformed if using normal score simulation

        Returns:
            True if data is acceptable for SGSIM
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
                f"Cannot run SGSIM simulation:\n\n{message}\n\n"
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

        # Check if using composited vs raw data
        if hasattr(self, 'data_source_raw') and self.data_source_raw.isChecked():
            dh_data = registry.get_drillhole_data(copy_data=False)
            if dh_data:
                has_composites = isinstance(dh_data.get('composites'), pd.DataFrame) and not dh_data.get('composites').empty
                if has_composites:
                    logger.warning(
                        "LINEAGE WARNING: Using raw assays for SGSIM when composites are available. "
                        "Raw assays have inconsistent sample support which violates change-of-support "
                        "principles for geostatistical simulation. Consider using composites."
                    )

        # Check for transformation metadata if required
        trans_meta = registry.get_transformation_metadata() if hasattr(registry, 'get_transformation_metadata') else None
        if trans_meta:
            logger.info(f"LINEAGE: Transformation metadata available for back-transformation")

        return True

    def run_analysis(self) -> None:
        """Override to add progress callback integration."""
        if not self.controller:
            self.show_warning("Unavailable", "Controller is not connected; cannot run analysis.")
            return
        
        if not self.validate_inputs():
            return
        
        # HARD GATE: Check data lineage before proceeding
        if not self._check_data_lineage():
            return
        
        params = self.gather_parameters()
        
        # Store simulation parameters for progress display
        self._total_realizations = params['nreal']
        self._last_progress_percent = 0
        self._last_progress_message = ""
        
        # Log start
        self._log_event("Starting SGSIM simulation...", "progress")
        self._log_event(f"  Variable: {self.variable}", "info")
        self._log_event(f"  Realizations: {params['nreal']}", "info")
        self._log_event(f"  Grid: {params['nx']}×{params['ny']}×{params['nz']}", "info")
        
        # Set up progress bar with simulation count
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0% - Initializing {params['nreal']} realizations...")
        
        self.show_progress("Running SGSIM...")

        def progress_callback(percent, message):
            """Update progress from worker thread using signals."""
            # Emit signal to update UI from main thread
            self.progress_updated.emit(percent, message)
        
        # Run with progress callback
        try:
            self.controller.run_sgsim(
                params=params,
                callback=self.handle_results,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Failed to dispatch SGSIM: {e}", exc_info=True)
            self._log_event(f"ERROR: {str(e)}", "error")
            self.hide_progress()

    def _find_renderer(self):
        """
        Find the renderer from various sources (main_window, controller, parent, QApplication).
        
        Returns:
            Renderer instance or None if not found.
        """
        renderer = None
        
        # Method 0: Direct main_window reference (set by MainWindow when creating panel)
        if hasattr(self, 'main_window') and self.main_window is not None:
            if hasattr(self.main_window, 'viewer_widget'):
                renderer = getattr(self.main_window.viewer_widget, 'renderer', None)
                if renderer:
                    logger.info("_find_renderer: Found via main_window.viewer_widget.renderer")
                    return renderer
            else:
                logger.debug("_find_renderer: main_window exists but no viewer_widget")
        else:
            logger.debug(f"_find_renderer: main_window not available (hasattr={hasattr(self, 'main_window')}, value={getattr(self, 'main_window', None)})")
        
        # Method 1: From controller directly (controller.r is the renderer)
        if self.controller:
            # AppController stores renderer as self.r, not self.viewer_widget
            renderer = getattr(self.controller, 'r', None)
            if renderer:
                logger.info("_find_renderer: Found via controller.r (direct renderer reference)")
                return renderer
            else:
                logger.debug("_find_renderer: controller exists but no 'r' attribute")
        
        # Method 2: From parent (if this is a child of MainWindow)
        try:
            parent = self.parent()
            if parent and hasattr(parent, 'viewer_widget'):
                renderer = getattr(parent.viewer_widget, 'renderer', None)
                if renderer:
                    logger.info("_find_renderer: Found via parent.viewer_widget.renderer")
                    return renderer
        except Exception as e:
            logger.debug(f"_find_renderer: Parent check failed: {e}")
        
        # Method 3: Walk up parent chain to find MainWindow
        try:
            widget = self
            for i in range(10):  # Limit iterations
                widget = widget.parent() if hasattr(widget, 'parent') else None
                if widget is None:
                    break
                if hasattr(widget, 'viewer_widget'):
                    renderer = getattr(widget.viewer_widget, 'renderer', None)
                    if renderer:
                        logger.info(f"_find_renderer: Found via parent chain (level {i})")
                        return renderer
        except Exception as e:
            logger.debug(f"_find_renderer: Parent chain walk failed: {e}")
        
        # Method 4: Try QApplication.instance() to find MainWindow
        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                for window in app.topLevelWidgets():
                    if hasattr(window, 'viewer_widget'):
                        renderer = getattr(window.viewer_widget, 'renderer', None)
                        if renderer:
                            logger.info(f"_find_renderer: Found via QApplication.topLevelWidgets ({type(window).__name__})")
                            return renderer
        except Exception as e:
            logger.debug(f"_find_renderer: QApplication check failed: {e}")
        
        logger.warning("_find_renderer: Could not find renderer via any method")
        return None
    
    def _log_event(self, message: str, level: str = "info"):
        """Add timestamped event to the log with color coding."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on level
        colors = {
            "info": "#e0e0e0",
            "success": "#81c784",
            "warning": "#ffb74d",
            "error": "#e57373",
            "progress": "#4fc3f7"
        }
        color = colors.get(level, "#e0e0e0")
        
        formatted = f'<span style="color: #888;">[{timestamp}]</span> <span style="color: {color};">{message}</span>'
        self.results_text.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self.results_text.verticalScrollBar()
        if scrollbar:
            scrollbar.setValue(scrollbar.maximum())
    
    def _update_progress(self, percent: int, message: str = ""):
        """Update progress bar and label with simulation count."""
        if not hasattr(self, 'progress_bar') or self.progress_bar is None:
            return
            
        percent = max(0, min(100, percent))
        
        # Ensure progress bar is visible
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        
        self.progress_bar.setValue(percent)
        
        # Format message for display
        if message:
            # Show message with percent in progress bar
            self.progress_bar.setFormat(f"{percent}% - {message}")
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.setText(message)
        else:
            self.progress_bar.setFormat(f"{percent}%")
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.setText(f"{percent}% complete")
        
        # Force UI repaint using safer processEvents (excludes user input to prevent reentrancy)
        # This ensures progress bar actually updates during long operations
        from PyQt6.QtCore import QEventLoop
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        
        # Log milestones and realization completions (throttled)
        # Log every 10% to avoid flooding the log
        should_log = (
            percent % 10 == 0 or 
            percent in [0, 100]
        )
        
        if should_log:
            self._log_event(f"Progress: {percent}% - {message}", "progress")
    
    def on_results(self, payload):
        self.sgsim_results = payload.get("results")
        self.results_ready = True
        
        # Store the full payload for accessing parameters
        self.sgsim_payload = payload
        
        # Update progress to 100%
        self._update_progress(100, "Complete!")
        
        # Get metadata for detailed logging
        metadata = payload.get("metadata", {})
        
        # Update Log with detailed information
        if self.sgsim_results and 'params' in self.sgsim_results:
            params = self.sgsim_results['params']
            n = params.nreal
            grid_dims = f"{params.nx}×{params.ny}×{params.nz}"
            n_blocks = params.nx * params.ny * params.nz
        else:
            n = self.nreal_spin.value()
            grid_dims = f"{self.nx.value()}×{self.ny.value()}×{self.nz.value()}"
            n_blocks = self.nx.value() * self.ny.value() * self.nz.value()
        
        self._log_event(f"✓ SGSIM COMPLETE", "success")
        self._log_event(f"  Realizations: {n}", "info")
        self._log_event(f"  Grid: {grid_dims} ({n_blocks:,} blocks)", "info")
        
        # =====================================================================
        # VALIDATION TABLE: Turns SGSIM from "black box" into defensible output
        # =====================================================================
        # These metrics validate that the simulation is statistically sound:
        # - Data variance: baseline uncertainty in the input data
        # - Mean conditional variance: average kriging variance (local uncertainty)
        # - Mean of SGSIM means: should be ~0 if Gaussian transform is correct
        # - Seed: for JORC/SAMREC reproducibility
        self._display_validation_table()
        
        # Log summary statistics if available
        summary = self.sgsim_results.get('summary', {}) if self.sgsim_results else {}
        if summary:
            mean_val = summary.get('mean')
            if mean_val is not None:
                mean_avg = np.nanmean(mean_val)
                std_val = summary.get('std')
                std_avg = np.nanstd(std_val) if std_val is not None else 0
                self._log_event(f"  Mean: {mean_avg:.3f}, Std: {std_avg:.3f}", "info")
        
        # =====================================================================
        # TRF-005 COMPLIANCE: Auto-trigger back-transformation
        # =====================================================================
        # Prevents users from accidentally using Gaussian values for metal/tonnage
        # calculations. Back-transform is performed automatically if a transformer
        # is available in the results.
        self._auto_back_transform()
        
        # Enable Controls (back_transform button may be disabled by auto-back-transform)
        self.export_btn.setEnabled(True)
        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90]:
            b.setEnabled(True)
            
        # Update Uncertainty Tab
        self._display_uncertainty()
        
        # Publish SGSIM results (include drillhole data so Resource panels can access it)
        if self.registry:
            try:
                # Add drillhole data to results for resource classification
                results_with_drillhole = dict(self.sgsim_results) if self.sgsim_results else {}
                if self.drillhole_data is not None and isinstance(self.drillhole_data, pd.DataFrame) and not self.drillhole_data.empty:
                    results_with_drillhole['drillhole_data'] = self.drillhole_data.copy()
                    logger.info(f"SGSIM: Including drillhole data ({len(self.drillhole_data)} points) in registry")
                
                self.registry.register_sgsim_results(results_with_drillhole, source_panel="SGSIM")
                self._log_event("Results registered to data registry", "info")
                
                # Also register the grid as a block model so Resource panels can detect it
                grid = self.sgsim_results.get('grid') or self.sgsim_results.get('pyvista_grid')
                if grid is not None:
                    try:
                        from ..models.block_model import BlockModel
                        import pyvista as pv
                        
                        if isinstance(grid, (pv.RectilinearGrid, pv.UnstructuredGrid, pv.StructuredGrid)):
                            if hasattr(grid, 'cell_centers'):
                                centers = grid.cell_centers()
                                if hasattr(centers, 'points'):
                                    coords = centers.points
                                    df_data = {
                                        'X': coords[:, 0],
                                        'Y': coords[:, 1],
                                        'Z': coords[:, 2]
                                    }
                                    if hasattr(grid, 'cell_data'):
                                        for key in grid.cell_data.keys():
                                            df_data[key] = grid.cell_data[key]
                                    
                                    df = pd.DataFrame(df_data)
                                    bm = BlockModel()
                                    bm.update_from_dataframe(df)
                                    self.registry.register_block_model_generated(
                                        bm, source_panel="SGSIM",
                                        metadata={"source": "sgsim_simulation"}
                                    )
                                    self._log_event("Block model registered for Resource panels", "info")
                    except Exception as e:
                        logger.warning(f"Failed to register SGSIM grid as block model: {e}")
            except AttributeError:
                logger.warning("register_sgsim_results not available in registry")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _on_sgsim_loaded(self, results: Dict[str, Any]):
        """
        Slot for DataRegistry.sgsimResultsLoaded.

        Ensures that if the user closes/reopens the SGSIM panel, the last
        simulation results are automatically restored, so they do not need
        to rerun SGSIM just to visualize or export.
        """
        self._restore_sgsim_results(results, source="signal")

    def _restore_sgsim_results(self, results: Dict[str, Any], source: str = "registry"):
        """Restore SGSIM results from DataRegistry into this panel's UI state."""
        if not results:
            return

        self.sgsim_results = results
        self.results_ready = True

        # Log basic info
        try:
            params = results.get("params")
            if params is not None:
                grid_dims = f"{params.nx}×{params.ny}×{params.nz}"
                n_blocks = params.nx * params.ny * params.nz
                n_real = getattr(params, "nreal", self.nreal_spin.value())
            else:
                grid_dims = "unknown"
                n_blocks = 0
                n_real = self.nreal_spin.value()

            self._log_event(f"✓ Restored SGSIM results ({source})", "success")
            self._log_event(f"  Realizations: {n_real}", "info")
            self._log_event(f"  Grid: {grid_dims} ({n_blocks:,} blocks)", "info")
        except Exception:
            # Best-effort logging; don't break restoration
            pass

        # Enable controls
        self.export_btn.setEnabled(True)
        self.back_transform_btn.setEnabled(True)
        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90]:
            b.setEnabled(True)

        # Refresh uncertainty tab and probability map buttons
        self._display_uncertainty()

    def _display_validation_table(self):
        """
        Display SGSIM validation metrics table.
        
        This turns SGSIM from "black box" into defensible engineering output
        by showing key validation statistics that prove the simulation is
        statistically sound and reproducible.
        
        Metrics displayed:
        - Data variance: Baseline uncertainty in the input data
        - Mean conditional variance: Average kriging variance (local uncertainty)
        - Mean of SGSIM means: Should be ~0 if Gaussian transform is correct
        - Realizations: Number of simulations run
        - Seed: Random seed for JORC/SAMREC reproducibility
        """
        if not self.sgsim_results:
            return
        
        # Extract validation metrics
        params = self.sgsim_results.get('params')
        summary = self.sgsim_results.get('summary', {})
        metadata = self.sgsim_results.get('metadata', {})
        
        # Data variance: variance of the input data
        data_variance = None
        if self.drillhole_data is not None and self.variable in self.drillhole_data.columns:
            clean_data = self.drillhole_data[self.variable].dropna()
            if len(clean_data) > 0:
                data_variance = np.var(clean_data)
        
        # Mean conditional variance: average kriging variance from simulations
        # If we have conditional_variance in results, use it. Otherwise compute from std.
        mean_cond_variance = None
        if 'conditional_variance' in self.sgsim_results:
            cond_var = self.sgsim_results['conditional_variance']
            mean_cond_variance = np.nanmean(cond_var)
        elif 'std' in summary:
            # Approximate: variance of the simulated values at each location
            # This is the average local uncertainty
            std = summary['std']
            mean_cond_variance = np.nanmean(std ** 2)
        
        # Mean of SGSIM means: should be ~0 if data was properly standardized
        mean_of_means = None
        if 'mean' in summary:
            mean_of_means = np.nanmean(summary['mean'])
        
        # Realizations count
        n_realizations = params.nreal if params else self.nreal_spin.value()
        
        # Random seed
        seed = metadata.get('seed') or params.seed if params else self.seed_spin.value()
        
        # Format the validation table
        self._log_event("", "info")  # Blank line for spacing
        self._log_event("═══════════════════════════════════════", "info")
        self._log_event("   SGSIM VALIDATION METRICS", "info")
        self._log_event("═══════════════════════════════════════", "info")
        
        if data_variance is not None:
            self._log_event(f"  Data variance:        {data_variance:.4f}", "info")
        
        if mean_cond_variance is not None:
            self._log_event(f"  Mean cond. variance:  {mean_cond_variance:.4f}", "info")
        
        if mean_of_means is not None:
            # Color code based on how close to zero
            if abs(mean_of_means) < 0.01:
                level = "success"
                indicator = "✓"
            elif abs(mean_of_means) < 0.05:
                level = "warning"
                indicator = "⚠"
            else:
                level = "error"
                indicator = "✗"
            
            self._log_event(f"  Mean of SGSIM means:  {mean_of_means:.4f} {indicator}", level)
        
        self._log_event(f"  Realizations:         {n_realizations}", "info")
        self._log_event(f"  Random seed:          {seed}", "info")
        self._log_event("═══════════════════════════════════════", "info")
        self._log_event("", "info")  # Blank line for spacing
        
        # Quality check warnings
        if mean_of_means is not None and abs(mean_of_means) > 0.05:
            self._log_event(
                "⚠ WARNING: Mean of SGSIM means is far from zero. "
                "This suggests the Gaussian transform may not be centered correctly. "
                "Review normal score transform or data preprocessing.",
                "warning"
            )
        
        if data_variance is not None and mean_cond_variance is not None:
            # Check that conditional variance is reasonable fraction of data variance
            ratio = mean_cond_variance / data_variance if data_variance > 0 else 0
            if ratio > 0.8:
                self._log_event(
                    f"⚠ WARNING: Mean conditional variance ({mean_cond_variance:.4f}) is {ratio*100:.1f}% "
                    f"of data variance ({data_variance:.4f}). This suggests low kriging efficiency. "
                    "Consider reviewing search parameters or data spacing.",
                    "warning"
                )
            elif ratio < 0.1:
                self._log_event(
                    f"✓ High kriging efficiency: conditional variance is {ratio*100:.1f}% of data variance",
                    "success"
                )

    def _display_uncertainty(self):
        if not self.sgsim_results:
            return
        exc = self.sgsim_results.get('exceedance', {})
        txt = "EXCEEDANCE VOLUMES:\n"
        for c, s in exc.items():
            txt += f"Cutoff {c}: Mean Tonnage {s.get('mean_tonnage', 0):,.0f}, Prob > 50%: {s.get('p50_tonnage', 0):,.0f}\n"
        self.uncert_text.setText(txt)
        
        # Buttons for Prob Maps
        # Clear existing
        while self.prob_layout.count():
            item = self.prob_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            
        maps = self.sgsim_results.get('probability_maps', {})
        for cut in maps.keys():
            b = QPushButton(f"> {cut}")
            b.clicked.connect(lambda checked, c=cut: self._visualize_probability(c))
            self.prob_layout.addWidget(b)

    def _visualize_summary(self, stat):
    """
    Visualize SGSIM summary statistics (mean, std, p10, p50, p90).

    CLEAN DESIGN:
    - Extract data from results (trust the simulation)
    - Create PyVista grid directly from metadata
    - Let renderer handle coordinate transforms
    - Async GPU handoff with 100ms delay
    """
    import numpy as np
    import pyvista as pv
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox

    self._log_event(f"Visualizing {stat.upper()}...", "progress")

    # 1. VALIDATE RESULTS
    if not self.sgsim_results:
        self._log_event("ERROR: No SGSIM results", "error")
        QMessageBox.warning(self, "No Results", "Run SGSIM simulation first.")
        return

    summary = self.sgsim_results.get('summary', {})
    if not summary:
        self._log_event("ERROR: No summary statistics", "error")
        QMessageBox.warning(self, "No Summary", "Summary statistics not found.")
        return

    # 2. GET STATISTIC DATA
    stat_map = {'mean': 'mean', 'std': 'std', 'p10': 'p10', 'p50': 'p50', 'p90': 'p90'}
    if stat not in stat_map:
        self._log_event(f"ERROR: Invalid statistic '{stat}'", "error")
        return

    stat_data = summary.get(stat_map[stat])
    if stat_data is None:
        self._log_event(f"ERROR: {stat} not in results", "error")
        QMessageBox.warning(self, "Not Available", f"{stat.upper()} not found in results.")
        return

    # 3. GET GRID PARAMETERS FROM METADATA
    metadata = self.sgsim_results.get('metadata', {})
    if not metadata:
        self._log_event("ERROR: No metadata in results", "error")
        QMessageBox.warning(self, "Metadata Missing", "Grid parameters not available.")
        return

    # Extract grid parameters (these come from the simulation itself)
    dims = metadata.get('grid_dims', (self.nx.value(), self.ny.value(), self.nz.value()))
    spacing = metadata.get('grid_spacing', (self.dx.value(), self.dy.value(), self.dz.value()))
    origin = metadata.get('grid_origin', (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()))

    nx, ny, nz = dims
    dx, dy, dz = spacing
    xmin, ymin, zmin = origin

    self._log_event(f"  Grid: {nx}×{ny}×{nz}, origin=({xmin:.1f}, {ymin:.1f}, {zmin:.1f})", "info")

    # 4. CREATE PYVISTA GRID
    try:
        # ImageData is memory-efficient and fast
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(dx, dy, dz),
            origin=(xmin, ymin, zmin)
        )

        # Flatten data (C-order) and assign to grid
        stat_flat = stat_data.flatten(order='C')

        # Property name format: ELEMENT_SGSIM_STAT (e.g., FE_PCT_SGSIM_MEAN)
        element = metadata.get('element', 'VALUE')
        property_name = f"{element}_SGSIM_{stat.upper()}"

        # Assign to cell data
        grid.cell_data[property_name] = stat_flat

        # Mark grid as pre-shifted if origin looks like local coords (< 1000)
        if abs(xmin) < 1000 and abs(ymin) < 1000:
            grid._coordinate_shifted = True

        n_cells = grid.n_cells
        self._log_event(f"  Created grid: {n_cells:,} cells, property={property_name}", "success")

        # 5. ASYNC GPU HANDOFF (100ms delay for thread stability)
        self._log_event(f"  → Sending {stat.upper()} to 3D viewer...", "progress")

        QTimer.singleShot(100, lambda: self._emit_visualization_safe(grid, property_name, stat))

    except Exception as e:
        self._log_event(f"ERROR: Grid creation failed: {e}", "error")
        QMessageBox.critical(self, "Grid Error", f"Failed to create grid:\n{e}")


    def _visualize_probability(self, cut):
    """
    Visualize SGSIM probability maps (Prob > cutoff).

    Same clean design as _visualize_summary.
    """
    import numpy as np
    import pyvista as pv
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QMessageBox

    self._log_event(f"Visualizing Prob > {cut}...", "progress")

    # 1. VALIDATE RESULTS
    if not self.sgsim_results:
        QMessageBox.warning(self, "No Results", "Run SGSIM simulation first.")
        return

    prob_maps = self.sgsim_results.get('probability_maps', {})
    prob_data = prob_maps.get(cut)

    if prob_data is None:
        QMessageBox.warning(self, "Not Available", f"Probability map for cutoff {cut} not found.")
        return

    # 2. GET GRID PARAMETERS FROM METADATA
    metadata = self.sgsim_results.get('metadata', {})
    if not metadata:
        QMessageBox.warning(self, "Metadata Missing", "Grid parameters not available.")
        return

    dims = metadata.get('grid_dims', (self.nx.value(), self.ny.value(), self.nz.value()))
    spacing = metadata.get('grid_spacing', (self.dx.value(), self.dy.value(), self.dz.value()))
    origin = metadata.get('grid_origin', (self.xmin_spin.value(), self.ymin_spin.value(), self.zmin_spin.value()))

    nx, ny, nz = dims
    dx, dy, dz = spacing
    xmin, ymin, zmin = origin

    # 3. CREATE PYVISTA GRID
    try:
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(dx, dy, dz),
            origin=(xmin, ymin, zmin)
        )

        # Flatten and assign probability data
        prob_flat = prob_data.flatten(order='C')

        element = metadata.get('element', 'VALUE')
        property_name = f"{element}_SGSIM_PROB_{cut}"

        grid.cell_data[property_name] = prob_flat

        # Mark as pre-shifted if needed
        if abs(xmin) < 1000 and abs(ymin) < 1000:
            grid._coordinate_shifted = True

        self._log_event(f"  → Sending Prob>{cut} to 3D viewer...", "progress")

        # Async handoff
        QTimer.singleShot(100, lambda: self._emit_visualization_safe(grid, property_name, f"Prob>{cut}"))

    except Exception as e:
        QMessageBox.critical(self, "Grid Error", f"Failed to create probability grid:\n{e}")


    def _emit_visualization_safe(self, grid, property_name, description):
    """
    Safe emission helper - catches errors during GPU handoff.

    Called via QTimer after 100ms delay to ensure:
    - Simulation thread has fully closed
    - Memory handles are released
    - UI thread is stable before GPU work
    """
    try:
        self.request_visualization.emit(grid, property_name)
        self._log_event(f"✓ {description} sent to 3D viewer", "success")
    except Exception as e:
        self._log_event(f"ERROR: Visualization failed: {e}", "error")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Visualization Failed", f"GPU handoff error:\n{e}")
                valid_data = self.drillhole_data[variable].dropna()
                input_mean = valid_data.mean()
                input_std = valid_data.std()
                input_var = valid_data.var()

                # Calculate reproduction errors
                mean_error = abs(sgsim_global_mean - input_mean)
                mean_bias_pct = (mean_error / input_mean) * 100 if input_mean != 0 else float('inf')

                var_error = abs(sgsim_global_var - input_var)
                var_bias_pct = (var_error / input_var) * 100 if input_var != 0 else float('inf')

                std_error = abs(sgsim_global_std - input_std)
                std_bias_pct = (std_error / input_std) * 100 if input_std != 0 else float('inf')

                # Log validation results
                self._log_event("✓ SGSIM Validation Results:", "info")
                self._log_event(f"  Input data:  mean={input_mean:.3f}, std={input_std:.3f}, var={input_var:.3f}", "info")
                self._log_event(f"  SGSIM grid: mean={sgsim_global_mean:.3f}, std={sgsim_global_std:.3f}, var={sgsim_global_var:.3f}", "info")
                self._log_event(f"  Mean bias:  {mean_error:.3f} ({mean_bias_pct:.1f}%)", "info")
                self._log_event(f"  Std bias:   {std_error:.3f} ({std_bias_pct:.1f}%)", "info")
                self._log_event(f"  Var bias:   {var_error:.3f} ({var_bias_pct:.1f}%)", "info")

                # Check for concerning bias levels
                warnings = []
                if mean_bias_pct > 5.0:  # >5% mean bias
                    warnings.append(f"• Mean bias: {mean_bias_pct:.1f}% (target: <5%)")
                if var_bias_pct > 15.0:  # >15% variance loss
                    warnings.append(f"• Variance bias: {var_bias_pct:.1f}% (target: <15%)")
                if std_bias_pct > 10.0:  # >10% std dev bias
                    warnings.append(f"• Std dev bias: {std_bias_pct:.1f}% (target: <10%)")

                if warnings:
                    warning_msg = (
                        f"⚠️ SGSIM Validation Warnings:\n\n" +
                        "\n".join(warnings) + "\n\n" +
                        "Recommendations:\n" +
                        "• Check that variogram was fitted on normal-score data\n" +
                        "• Verify neighbourhood settings (min 8, max 16 recommended)\n" +
                        "• Consider adjusting variogram parameters\n" +
                        "• Review data transformation and declustering"
                    )
                    self._log_event("WARNING: SGSIM statistics show concerning bias levels", "warning")
                    QMessageBox.warning(self, "SGSIM Validation Warning", warning_msg)
                else:
                    self._log_event("✓ SGSIM statistics show good reproduction of input data", "success")

        except Exception as e:
            self._log_event(f"Warning: Could not perform SGSIM validation: {str(e)}", "warning")
            logger.warning(f"SGSIM validation error: {e}")

    # ------------------------------------------------------------------
    # Reblocking helper
    # ------------------------------------------------------------------
    def _reblock_for_visualization(self, data: Any, params: "SGSIMParameters"):
        """
        Optionally reblock SGSIM summary/probability grids to a coarser
        visualization block size using simple block means.

        Returns (data_out, params_out). If reblocking is disabled or invalid,
        returns original (data, params).
        """
        import numpy as np

        # If checkbox is off, use simulation grid directly
        if not getattr(self, "viz_reblock_check", None) or not self.viz_reblock_check.isChecked():
            return data, params

        # Ensure we have numeric array
        arr = np.asarray(data)
        if arr.ndim != 3:
            self._log_event("Reblock skipped: data is not 3D", "warning")
            return data, params

        sim_dx, sim_dy, sim_dz = float(params.xinc), float(params.yinc), float(params.zinc)
        viz_dx, viz_dy, viz_dz = float(self.viz_dx.value()), float(self.viz_dy.value()), float(self.viz_dz.value())

        # If viz sizes match sim sizes within tolerance, no reblock needed
        tol = 1e-6
        if (
            abs(viz_dx - sim_dx) < tol
            and abs(viz_dy - sim_dy) < tol
            and abs(viz_dz - sim_dz) < tol
        ):
            return data, params

        # Factors must be >=1 and near integer multiples
        fx = viz_dx / sim_dx
        fy = viz_dy / sim_dy
        fz = viz_dz / sim_dz

        fx_i, fy_i, fz_i = int(round(fx)), int(round(fy)), int(round(fz))
        if (
            fx_i < 1
            or fy_i < 1
            or fz_i < 1
            or abs(fx - fx_i) > 1e-3
            or abs(fy - fy_i) > 1e-3
            or abs(fz - fz_i) > 1e-3
        ):
            self._log_event(
                "Reblock skipped: visualization block sizes must be integer multiples of simulation DX/DY/DZ",
                "warning",
            )
            return data, params

        nz, ny, nx = arr.shape
        nz_new = nz // fz_i
        ny_new = ny // fy_i
        nx_new = nx // fx_i

        if nz_new < 1 or ny_new < 1 or nx_new < 1:
            self._log_event("Reblock skipped: factors too large for grid size", "warning")
            return data, params

        # Trim edges if not divisible
        trim = False
        if nz_new * fz_i != nz or ny_new * fy_i != ny or nx_new * fx_i != nx:
            trim = True
        if trim:
            self._log_event(
                "Reblock: trimming edge cells so grid is divisible by reblock factors",
                "warning",
            )
        arr_trim = arr[: nz_new * fz_i, : ny_new * fy_i, : nx_new * fx_i]

        # Reshape and average: (nz_new, fz_i, ny_new, fy_i, nx_new, fx_i) → mean over small blocks
        arr_rb = (
            arr_trim.reshape(nz_new, fz_i, ny_new, fy_i, nx_new, fx_i)
            .mean(axis=(1, 3, 5))
        )

        from ..models.sgsim3d import SGSIMParameters

        viz_params = SGSIMParameters(
            nreal=params.nreal,
            nx=nx_new,
            ny=ny_new,
            nz=nz_new,
            xmin=params.xmin,
            ymin=params.ymin,
            zmin=params.zmin,
            xinc=viz_dx,
            yinc=viz_dy,
            zinc=viz_dz,
            variogram_type=params.variogram_type,
            range_major=params.range_major,
            range_minor=params.range_minor,
            range_vert=params.range_vert,
            azimuth=params.azimuth,
            dip=params.dip,
            nugget=params.nugget,
            sill=params.sill,
            min_neighbors=params.min_neighbors,
            max_neighbors=params.max_neighbors,
            max_search_radius=params.max_search_radius,
            seed=params.seed,
            parallel=params.parallel,
            n_jobs=params.n_jobs,
            method=params.method,
            use_numba=params.use_numba,
        )

        self._log_event(
            f"Reblocked for visualization: {params.nx}×{params.ny}×{params.nz} → "
            f"{viz_params.nx}×{viz_params.ny}×{viz_params.nz} cells "
            f"(DX/DY/DZ = {viz_dx:.2f}/{viz_dy:.2f}/{viz_dz:.2f} m)",
            "info",
        )

        return arr_rb, viz_params
    
    def _load_from_variogram(self):
        # Search parent
        mw = self.parent()
        if hasattr(mw, 'variogram_dialog') and mw.variogram_dialog:
            # Use get_variogram_results() method if available, otherwise access attribute directly
            if hasattr(mw.variogram_dialog, 'get_variogram_results'):
                res = mw.variogram_dialog.get_variogram_results()
            else:
                res = getattr(mw.variogram_dialog, 'variogram_results', None)
            if res:
                self.set_variogram_results(res)
        elif self.variogram_results:
            # Use cached results
            self.set_variogram_results(self.variogram_results)

    def clear_results(self):
        self.sgsim_results = None
        self.results_ready = False
        self.results_text.clear()
        self.uncert_text.clear()
        # Disable buttons
        for b in [self.viz_mean, self.viz_std, self.viz_p10, self.viz_p50, self.viz_p90, self.export_btn, self.back_transform_btn]:
            b.setEnabled(False)
        # Clear probability buttons
        while self.prob_layout.count():
            item = self.prob_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        # Clear internal state
        self.drillhole_data = None
        self.variogram_results = None
        self.transformation_metadata = None

        # Clear results (reuse existing method)
        self.clear_results()

        # Clear matplotlib canvases if they exist
        for canvas_name in ['canvas', 'canvas_2d', 'canvas_3d']:
            canvas = getattr(self, canvas_name, None)
            if canvas and hasattr(canvas, 'figure'):
                try:
                    canvas.figure.clear()
                    canvas.draw()
                except Exception:
                    pass

        # Call base class to clear common widgets
        super().clear_panel()
        logger.info("SGSIMPanel: Panel fully cleared")

    def _export_results_menu(self):
        """Export SGSIM results to file."""
        if not self.sgsim_results:
            QMessageBox.warning(self, "No Results", "Run simulation first.")
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
            "Export SGSIM Results",
            f"sgsim_results.{extension}",
            f"{extension.upper()} Files (*.{extension})"
        )

        if not file_path:
            return

        try:
            # Import export functions
            from ..models.geostat_export import export_sgsim_results_to_csv, export_sgsim_results_to_vtk
            from ..models.geostat_results import SGSIMResults

            # Reconstruct SGSIMResults object from dictionary data
            summary = self.sgsim_results.get('summary', {})

            # Create SGSIMResults object
            # Map summary fields to SGSIMResults field names
            mean_data = summary.get('mean')
            std_data = summary.get('std')
            cv_data = summary.get('cv')

            if mean_data is None or std_data is None:
                raise ValueError("Required summary statistics (mean, std) not found in SGSIM results")

            variance_data = summary.get('var')  # Note: summary uses 'var', not 'variance'
            if variance_data is None:
                # Compute variance from std if var not available
                variance_data = std_data ** 2

            if cv_data is None:
                # Compute coefficient of variation if not available
                with np.errstate(divide='ignore', invalid='ignore'):
                    cv_data = std_data / mean_data
                    cv_data[mean_data == 0] = 0

            sgsim_results_obj = SGSIMResults(
                realizations=self.sgsim_results.get('realizations_raw'),
                mean=mean_data,
                variance=variance_data,
                std_dev=std_data,  # Note: summary uses 'std', not 'std_dev'
                coefficient_of_variation=cv_data,  # Note: summary uses 'cv', not 'coefficient_of_variation'
                p10=summary.get('p10'),
                p50=summary.get('p50'),
                p90=summary.get('p90'),
                probability_above_cutoff=summary.get('probability_above_cutoff'),
                metadata=self.sgsim_results.get('params', {}).__dict__ if hasattr(self.sgsim_results.get('params'), '__dict__') else {}
            )

            # Get grid parameters
            params = self.sgsim_results.get('params')
            if params is None:
                raise ValueError("Grid parameters not found in results")

            if format_type == 'csv':
                # Create coordinate arrays for CSV export
                nx, ny, nz = params.nx, params.ny, params.nz
                xinc, yinc, zinc = params.xinc, params.yinc, params.zinc
                xmin, ymin, zmin = params.xmin, params.ymin, params.zmin

                # Create 1D coordinate arrays
                x_coords = np.linspace(xmin, xmin + (nx-1)*xinc, nx)
                y_coords = np.linspace(ymin, ymin + (ny-1)*yinc, ny)
                z_coords = np.linspace(zmin, zmin + (nz-1)*zinc, nz)

                # Create 3D coordinate grids for all points (matching SGSIM data layout)
                # SGSIM data is (nz, ny, nx), so we need meshgrid with z,y,x ordering
                X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

                # Export to CSV
                export_sgsim_results_to_csv(sgsim_results_obj, coords, file_path)

            else:  # VTK format
                # Create 3D grid coordinate arrays for VTK export (matching PyVista StructuredGrid expectations)
                nx, ny, nz = params.nx, params.ny, params.nz
                xinc, yinc, zinc = params.xinc, params.yinc, params.zinc
                xmin, ymin, zmin = params.xmin, params.ymin, params.zmin

                # Create 1D coordinate arrays
                x_coords = np.linspace(xmin, xmin + (nx-1)*xinc, nx)
                y_coords = np.linspace(ymin, ymin + (ny-1)*yinc, ny)
                z_coords = np.linspace(zmin, zmin + (nz-1)*zinc, nz)

                # Create 3D coordinate grids (nx, ny, nz) as expected by PyVista StructuredGrid
                grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

                # Export to VTK
                export_sgsim_results_to_vtk(sgsim_results_obj, grid_x, grid_y, grid_z, file_path)

            QMessageBox.information(
                self,
                "Export Complete",
                f"✓ SGSIM results exported successfully!\n\nFile: {file_path}\nFormat: {format_type.upper()}"
            )

        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Failed to export SGSIM results:\n{str(e)}")

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
            
            # Simulation parameters
            settings['nreal'] = get_safe_widget_value(self, 'nreal_spin')
            settings['seed'] = get_safe_widget_value(self, 'seed_spin')
            
            # Variogram model
            settings['model_type'] = get_safe_widget_value(self, 'model_combo')
            settings['nugget'] = get_safe_widget_value(self, 'nugget_spin')
            settings['sill'] = get_safe_widget_value(self, 'sill_spin')
            settings['range'] = get_safe_widget_value(self, 'range_spin')
            
            # Anisotropy
            settings['azimuth'] = get_safe_widget_value(self, 'azimuth_spin')
            settings['dip'] = get_safe_widget_value(self, 'dip_spin')
            
            # Grid
            settings['xmin'] = get_safe_widget_value(self, 'xmin_spin')
            settings['ymin'] = get_safe_widget_value(self, 'ymin_spin')
            settings['zmin'] = get_safe_widget_value(self, 'zmin_spin')
            settings['grid_x'] = get_safe_widget_value(self, 'dx_spin')
            settings['grid_y'] = get_safe_widget_value(self, 'dy_spin')
            settings['grid_z'] = get_safe_widget_value(self, 'dz_spin')
            settings['nx'] = get_safe_widget_value(self, 'nx_spin')
            settings['ny'] = get_safe_widget_value(self, 'ny_spin')
            settings['nz'] = get_safe_widget_value(self, 'nz_spin')
            
            # Search settings
            settings['neighbors'] = get_safe_widget_value(self, 'neighbors_spin')
            
            # Post-processing
            settings['cutoff'] = get_safe_widget_value(self, 'cutoff_spin')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save SGSIM panel settings: {e}")
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
            
            # Simulation parameters
            set_safe_widget_value(self, 'nreal_spin', settings.get('nreal'))
            set_safe_widget_value(self, 'seed_spin', settings.get('seed'))
            
            # Variogram model
            set_safe_widget_value(self, 'model_combo', settings.get('model_type'))
            set_safe_widget_value(self, 'nugget_spin', settings.get('nugget'))
            set_safe_widget_value(self, 'sill_spin', settings.get('sill'))
            set_safe_widget_value(self, 'range_spin', settings.get('range'))
            
            # Anisotropy
            set_safe_widget_value(self, 'azimuth_spin', settings.get('azimuth'))
            set_safe_widget_value(self, 'dip_spin', settings.get('dip'))
            
            # Grid
            set_safe_widget_value(self, 'xmin_spin', settings.get('xmin'))
            set_safe_widget_value(self, 'ymin_spin', settings.get('ymin'))
            set_safe_widget_value(self, 'zmin_spin', settings.get('zmin'))
            set_safe_widget_value(self, 'dx_spin', settings.get('grid_x'))
            set_safe_widget_value(self, 'dy_spin', settings.get('grid_y'))
            set_safe_widget_value(self, 'dz_spin', settings.get('grid_z'))
            set_safe_widget_value(self, 'nx_spin', settings.get('nx'))
            set_safe_widget_value(self, 'ny_spin', settings.get('ny'))
            set_safe_widget_value(self, 'nz_spin', settings.get('nz'))
            
            # Search settings
            set_safe_widget_value(self, 'neighbors_spin', settings.get('neighbors'))
            
            # Post-processing
            set_safe_widget_value(self, 'cutoff_spin', settings.get('cutoff'))
                
            logger.info("Restored SGSIM panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore SGSIM panel settings: {e}")
