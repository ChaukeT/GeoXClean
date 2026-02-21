"""
GRADE TRANSFORMATION PANEL

Purpose: Apply mathematical transformations (Log, Box-Cox, Normal Score) to grade data.

Integration: Uses 'transform.py' for Normal Score to ensure back-transform compatibility.

AUDIT COMPLIANCE:
- TRF-001: Lineage gate warning for raw assays
- TRF-004: Full provenance metadata in transformations dict
- TRF-008: Transform history tracking (prevents stacked transforms)
- TRF-009: Value replacement warnings for Log/Sqrt/Box-Cox
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QFormLayout, QDoubleSpinBox, QCheckBox, QSplitter,
    QFileDialog, QFrame, QStyle, QTabWidget
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel
# Import the robust transformer and filter utility
from ..models.transform import NormalScoreTransformer, TRANSFORM_SUFFIXES, filter_transformed_columns

# Optional Dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    # Matplotlib backend is set in main.py
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    NavigationToolbar = None

from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, ModernColors

logger = logging.getLogger(__name__)


class GradeTransformationPanel(BaseAnalysisPanel):
    task_name = "grade_transform"

    # PanelManager metadata
    PANEL_ID = "GradeTransformationPanel"
    PANEL_NAME = "GradeTransformation Panel"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent=parent, panel_id="grade_transform")
        self.setWindowTitle("Grade Data Transformation")
        self.resize(1200, 800)
        self.setStyleSheet(get_analysis_panel_stylesheet())
        
        self.main_window = main_window
        
        # State
        self.original_data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.transformations: Dict[str, Dict] = {}  # Metadata with provenance (TRF-004)
        self.active_transformers: Dict[str, NormalScoreTransformer] = {}  # Store NST objects
        self.registry_snapshot: Optional[Dict] = None  # Keep full registry data reference
        self._composites: Optional[pd.DataFrame] = None
        self._assays: Optional[pd.DataFrame] = None
        
        # TRF-008: Transform history tracking
        self._transform_history: List[Dict[str, Any]] = []

        # TRF-001: Track if user was warned about raw assays
        self._raw_assay_warning_shown: bool = False

        # Preview state for export functionality
        self._preview_col: Optional[str] = None
        self._preview_method: Optional[str] = None
        self._preview_trans_vals: Optional[np.ndarray] = None

        # Track if data has changed since last view (must init BEFORE _connect_registry)
        self._pending_data_update = False
        self._ui_ready = False

        self._connect_registry()
        self._build_ui()
        self._ui_ready = True

    def _connect_registry(self):
        """Connect to the DataRegistry.
        
        FIX: Added null check for signal property before calling .connect().
        The drillholeDataLoaded property returns None if _signals is not initialized.
        """
        try:
            self.registry = self.get_registry()
            if not self.registry:
                logger.warning("DataRegistry not available - get_registry() returned None")
                return
            
            # FIX: Check if signal is available before connecting
            signal = self.registry.drillholeDataLoaded
            if signal is not None:
                signal.connect(self._on_data_loaded)
                logger.debug("GradeTransformationPanel: Connected to drillholeDataLoaded signal")
            else:
                logger.warning("GradeTransformationPanel: drillholeDataLoaded signal not available")
            
            # Load existing if available (handles lazy panel creation case)
            existing = self.registry.get_drillhole_data()
            if existing:
                self._on_data_loaded(existing)
        except Exception as exc:
            logger.warning(f"DataRegistry connection failed: {exc}", exc_info=True)
            self.registry = None

    def _build_ui(self):
        layout = self.main_layout
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Header
        header = QFrame()
        header.setStyleSheet(f"background-color: #333; border-bottom: 1px solid {ModernColors.PANEL_BG};")
        header.setFixedHeight(50)
        hl = QHBoxLayout(header)
        hl.addWidget(QLabel("Grade Transformation"))
        hl.addStretch()
        
        self.data_source_combo = QComboBox()
        self.data_source_combo.setFixedWidth(200)
        self.data_source_combo.addItems(["Composited Data", "Raw Assays"])
        self.data_source_combo.currentTextChanged.connect(self._on_source_changed)
        hl.addWidget(QLabel("Source:"))
        hl.addWidget(self.data_source_combo)

        # Refresh button to get latest data from registry
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setFixedWidth(80)
        self.refresh_btn.setToolTip("Refresh data from registry (get latest composites/assays)")
        self.refresh_btn.setStyleSheet("""
            QPushButton { background-color: #2a2a3a; border: 1px solid #3a3a4a; border-radius: 4px; padding: 4px 8px; }
            QPushButton:hover { background-color: #3a3a4a; border-color: #4a9eff; }
        """)
        self.refresh_btn.clicked.connect(self._manual_refresh)
        hl.addWidget(self.refresh_btn)
        layout.addWidget(header)

        # New data notification banner (hidden by default)
        self._new_data_banner = QLabel("🔔 New data available! Click 'Refresh' to load the latest composites.")
        self._new_data_banner.setStyleSheet("""
            QLabel {
                background-color: #1a3a5a;
                color: #4fc3f7;
                padding: 8px;
                border: 1px solid #2196F3;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        self._new_data_banner.setVisible(False)
        layout.addWidget(self._new_data_banner)

        # TRF-001: Warning banner for raw assays
        self.raw_assay_warning = QLabel(
            "⚠️ WARNING: Transforming raw assays violates change-of-support principles.\n"
            "For JORC/SAMREC compliance, use composited data. Raw assays are for exploratory analysis only."
        )
        self.raw_assay_warning.setObjectName("WarningLabel")
        self.raw_assay_warning.setWordWrap(True)
        self.raw_assay_warning.setVisible(False)
        layout.addWidget(self.raw_assay_warning)

        # 2. Content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Controls
        left = QWidget()
        l_layout = QVBoxLayout(left)
        
        # Column
        gb_col = QGroupBox("1. Select Column")
        f_col = QFormLayout()
        f_col.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_col.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.col_combo = QComboBox()
        self.col_combo.currentTextChanged.connect(self._on_col_changed)
        self.col_combo.setToolTip(
            "Select the grade column to transform.\n"
            "Transformations normalize skewed distributions for kriging.\n"
            "Common for gold, copper, and other high-coefficient-of-variation grades."
        )
        f_col.addRow("Grade:", self.col_combo)
        
        # TRF-008: Show if column already transformed
        self.transform_status_lbl = QLabel("")
        self.transform_status_lbl.setStyleSheet("color: #888; font-style: italic;")
        f_col.addRow("Status:", self.transform_status_lbl)
        
        gb_col.setLayout(f_col)
        l_layout.addWidget(gb_col)
        
        # Transform
        gb_trans = QGroupBox("2. Transformation")
        f_trans = QFormLayout()
        f_trans.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        f_trans.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.trans_combo = QComboBox()
        self.trans_combo.addItems(["None", "Log", "Log10", "Square Root", "Box-Cox", "Normal Score"])
        self.trans_combo.currentTextChanged.connect(self._on_method_changed)
        self.trans_combo.setToolTip(
            "Transformation method:\n"
            "• None: No transformation (use for Gaussian data)\n"
            "• Log: Natural logarithm (ln) - for lognormal data\n"
            "• Log10: Base-10 logarithm - for strongly skewed data\n"
            "• Square Root: √x - mild transformation for moderately skewed data\n"
            "• Box-Cox: Optimal power transform (finds best λ parameter)\n"
            "• Normal Score: Rank-based Gaussian transform (robust, preserves spatial structure)"
        )
        f_trans.addRow("Method:", self.trans_combo)

        # Params
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(-5, 5)
        self.lambda_spin.setToolTip(
            "Box-Cox lambda parameter:\n"
            "• λ = 1: No transformation\n"
            "• λ = 0.5: Square root\n"
            "• λ = 0: Log transformation\n"
            "• λ = -1: Reciprocal\n\n"
            "Optimal λ is auto-calculated to maximize normality."
        )
        self.lambda_lbl = QLabel("Lambda:")
        f_trans.addRow(self.lambda_lbl, self.lambda_spin)

        self.shift_chk = QCheckBox("Add Constant")
        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(0, 1000)
        self.shift_spin.setValue(0.01)
        self.shift_spin.setToolTip(
            "Constant added before transformation to handle zero/negative values.\n"
            "Typical: 0.01 (small positive value).\n"
            "Required for Log, Log10, and Box-Cox with zeros."
        )
        h_shift = QHBoxLayout()
        h_shift.addWidget(self.shift_chk)
        h_shift.addWidget(self.shift_spin)
        f_trans.addRow("Zero Handling:", h_shift)
        
        # TRF-002: Option to use declustering weights
        self.use_weights_chk = QCheckBox("Use Declustering Weights")
        self.use_weights_chk.setToolTip(
            "If declustering has been performed, use the weights to compute\n"
            "the transformation CDF. This prevents clustered samples from\n"
            "biasing the global distribution."
        )
        f_trans.addRow("Declustering:", self.use_weights_chk)
        
        gb_trans.setLayout(f_trans)
        l_layout.addWidget(gb_trans)
        
        # Actions
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.clicked.connect(self._preview)
        l_layout.addWidget(self.btn_preview)
        
        self.btn_apply = QPushButton("Apply & Create Column")
        self.btn_apply.setObjectName("PrimaryButton")
        self.btn_apply.clicked.connect(self._apply)
        l_layout.addWidget(self.btn_apply)
        
        l_layout.addStretch()
        
        # Transform history display
        gb_history = QGroupBox("Transform History")
        h_layout = QVBoxLayout()
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Column", "Method", "New Column", "Source"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setMaximumHeight(150)
        h_layout.addWidget(self.history_table)
        gb_history.setLayout(h_layout)
        l_layout.addWidget(gb_history)
        
        self.btn_send = QPushButton("Update Registry")
        self.btn_send.setObjectName("PrimaryButton")
        self.btn_send.clicked.connect(self._send_to_registry)
        l_layout.addWidget(self.btn_send)
        
        splitter.addWidget(left)

        # Right: Viz
        right = QTabWidget()

        # Tab 1: Plot with interactive toolbar
        self.tab_plot = QWidget()
        p_layout = QVBoxLayout(self.tab_plot)
        if MATPLOTLIB_AVAILABLE:
            # Toolbar container for navigation + export
            toolbar_container = QWidget()
            toolbar_layout = QHBoxLayout(toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            self.chart_toolbar_layout = toolbar_layout

            # Export chart button
            self.chart_export_btn = QPushButton("Export Chart")
            self.chart_export_btn.setToolTip("Export chart as PNG, SVG, or PDF")
            self.chart_export_btn.clicked.connect(self._export_chart)
            self.chart_export_btn.setEnabled(False)
            self.chart_export_btn.setStyleSheet("""
                QPushButton { background-color: #005a9e; padding: 4px 12px; }
                QPushButton:hover { background-color: #0065b3; }
                QPushButton:disabled { background-color: #444; color: #888; }
            """)
            toolbar_layout.addStretch()
            toolbar_layout.addWidget(self.chart_export_btn)

            p_layout.addWidget(toolbar_container)

            # Canvas
            self.canvas = FigureCanvas(Figure(figsize=(5, 4), facecolor='#2b2b2b'))
            p_layout.addWidget(self.canvas)

            # Navigation toolbar (interactive zoom/pan)
            self.nav_toolbar = None
            if NavigationToolbar is not None:
                self.nav_toolbar = NavigationToolbar(self.canvas, self.tab_plot)
                self.nav_toolbar.setStyleSheet("background-color: #333;")
                p_layout.addWidget(self.nav_toolbar)
        else:
            self.chart_export_btn = None
            self.nav_toolbar = None
            p_layout.addWidget(QLabel("Matplotlib required for plots"))
        right.addTab(self.tab_plot, "Distribution")

        # Tab 2: Data Preview with full data and export
        self.tab_data_widget = QWidget()
        data_layout = QVBoxLayout(self.tab_data_widget)

        # Data preview header with stats and export
        data_header = QHBoxLayout()
        self.data_stats_lbl = QLabel("No data loaded")
        self.data_stats_lbl.setStyleSheet("color: #aaa; font-style: italic;")
        data_header.addWidget(self.data_stats_lbl)
        data_header.addStretch()

        self.data_export_btn = QPushButton("Export Data (CSV)")
        self.data_export_btn.setToolTip("Export full transformed data to CSV")
        self.data_export_btn.clicked.connect(self._export_data)
        self.data_export_btn.setEnabled(False)
        self.data_export_btn.setStyleSheet("""
            QPushButton { background-color: #005a9e; padding: 4px 12px; }
            QPushButton:hover { background-color: #0065b3; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """)
        data_header.addWidget(self.data_export_btn)
        data_layout.addLayout(data_header)

        # Summary statistics group
        self.stats_group = QGroupBox("Summary Statistics")
        stats_layout = QHBoxLayout()
        self.orig_stats_lbl = QLabel("Original: -")
        self.orig_stats_lbl.setWordWrap(True)
        self.trans_stats_lbl = QLabel("Transformed: -")
        self.trans_stats_lbl.setWordWrap(True)
        stats_layout.addWidget(self.orig_stats_lbl)
        stats_layout.addWidget(self.trans_stats_lbl)
        self.stats_group.setLayout(stats_layout)
        data_layout.addWidget(self.stats_group)

        # Data table
        self.tab_data = QTableWidget()
        self.tab_data.setStyleSheet(f"""
            QTableWidget {{ background-color: {ModernColors.PANEL_BG}; gridline-color: #333; }}
            QHeaderView::section {{ background-color: #333; color: #ccc; padding: 4px; }}
        """)
        data_layout.addWidget(self.tab_data)

        right.addTab(self.tab_data_widget, "Data Preview")

        splitter.addWidget(right)
        splitter.setSizes([400, 800])
        layout.addWidget(splitter)

        self._on_method_changed("None")

    # =========================================================
    # LOGIC
    # =========================================================

    def _on_data_loaded(self, data):
        """Called when registry updates.

        Shows notification banner if panel is visible, otherwise marks for refresh.
        """
        if not isinstance(data, dict):
            return

        self.registry_snapshot = data  # Keep full ref
        self._composites = data.get('composites')
        self._assays = data.get('assays')

        # If panel is visible, show notification banner (user decides when to refresh)
        # If panel is hidden, auto-update on next show
        if self.isVisible():
            # Show notification banner - let user decide when to refresh
            if hasattr(self, '_new_data_banner'):
                self._new_data_banner.setVisible(True)
                self._pending_data_update = True
                logger.info("GradeTransformationPanel: New data available, notification shown")
        else:
            # Panel not visible - auto-update when shown
            self._pending_data_update = True
            self._apply_data_update()

    def _apply_data_update(self):
        """Apply the pending data update to UI controls."""
        # Only apply if UI is ready
        if not getattr(self, '_ui_ready', False):
            return

        # Auto-select (only if UI is built)
        if hasattr(self, 'data_source_combo') and self.data_source_combo is not None:
            self.data_source_combo.blockSignals(True)
            if self._composites is not None and not self._composites.empty:
                self.data_source_combo.setCurrentIndex(0)
            elif self._assays is not None:
                self.data_source_combo.setCurrentIndex(1)
            self.data_source_combo.blockSignals(False)

        self._on_source_changed()

        # Hide notification banner
        if hasattr(self, '_new_data_banner'):
            self._new_data_banner.setVisible(False)
        self._pending_data_update = False

    def _manual_refresh(self):
        """Manual refresh - reload data from registry."""
        try:
            registry = self.get_registry()
            if registry:
                data = registry.get_drillhole_data()
                if data:
                    self.registry_snapshot = data
                    self._composites = data.get('composites')
                    self._assays = data.get('assays')
                    self._apply_data_update()

                    # Show feedback
                    comp_count = len(self._composites) if self._composites is not None else 0
                    assay_count = len(self._assays) if self._assays is not None else 0
                    logger.info(f"GradeTransformationPanel: Refreshed - {comp_count} composites, {assay_count} assays")
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}", exc_info=True)

    def showEvent(self, event):
        """Auto-refresh when panel becomes visible."""
        super().showEvent(event)

        # If there's a pending update, apply it now
        if getattr(self, '_pending_data_update', False):
            self._apply_data_update()

    def _on_source_changed(self):
        """Handle data source selection change."""
        if not hasattr(self, 'data_source_combo') or self.data_source_combo is None:
            return
        txt = self.data_source_combo.currentText()
        
        # =====================================================================
        # TRF-001 COMPLIANCE: Lineage gate for raw assays
        # =====================================================================
        if "Raw" in txt:
            # Show persistent warning banner
            self.raw_assay_warning.setVisible(True)
            
            # Show modal warning if not already shown this session
            if not self._raw_assay_warning_shown:
                self._raw_assay_warning_shown = True
                logger.warning("TRF-001 LINEAGE: User selected raw assays for transformation")
                
                reply = QMessageBox.warning(
                    self,
                    "Lineage Warning - Raw Assays Selected",
                    "⚠️ CHANGE-OF-SUPPORT VIOLATION\n\n"
                    "Transforming raw assays violates geostatistical best practices:\n\n"
                    "• Raw assays have inconsistent sample support (variable lengths)\n"
                    "• The resulting variogram will be biased\n"
                    "• JORC/SAMREC defensibility may be compromised\n\n"
                    "RECOMMENDATION: Use composited data for production workflows.\n\n"
                    "Do you want to proceed with raw assays (for exploratory analysis)?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    # Revert to composites if available
                    if self._composites is not None and not self._composites.empty:
                        self.data_source_combo.blockSignals(True)
                        self.data_source_combo.setCurrentIndex(0)
                        self.data_source_combo.blockSignals(False)
                        self.raw_assay_warning.setVisible(False)
                        txt = "Composited Data"
                    else:
                        logger.info("User forced to use raw assays (no composites available)")
        else:
            self.raw_assay_warning.setVisible(False)
        
        # Set data based on selection
        if "Composited" in txt and self._composites is not None:
            self.original_data = self._composites.copy()
        elif "Raw" in txt and self._assays is not None:
            self.original_data = self._assays.copy()
        else:
            self.original_data = None
            
        self.transformed_data = self.original_data.copy() if self.original_data is not None else None
        self._populate_columns()
        
        # Check for declustering weights availability
        self._update_weights_availability()

    def _update_weights_availability(self):
        """Check if declustering weights are available and update UI."""
        has_weights = False
        
        if self.original_data is not None:
            # Check for DECLUST_WEIGHT column
            weight_cols = [c for c in self.original_data.columns if 'WEIGHT' in c.upper() and 'DECLUST' in c.upper()]
            has_weights = len(weight_cols) > 0
        
        self.use_weights_chk.setEnabled(has_weights)
        if not has_weights:
            self.use_weights_chk.setChecked(False)
            self.use_weights_chk.setToolTip(
                "Declustering weights not available.\n"
                "Run declustering first to enable weighted transformation."
            )
        else:
            self.use_weights_chk.setToolTip(
                "Use declustering weights for CDF calculation.\n"
                "This prevents clustered samples from biasing the transformation."
            )

    def _populate_columns(self):
        """Populate column dropdown with numeric grade columns."""
        self.col_combo.clear()
        if self.original_data is None:
            return
        
        # Numeric only, exclude coords, system IDs, and compositing metadata
        nums = self.original_data.select_dtypes(include=[np.number]).columns
        ignore = {
            'X', 'Y', 'Z', 'FROM', 'TO', 'LENGTH', 'HOLEID', 'MID', 'ELEVATION', 'GLOBAL_INTERVAL_ID',
            # Compositing metadata columns (not geological properties)
            'SAMPLE_COUNT', 'TOTAL_MASS', 'TOTAL_LENGTH', 'SUPPORT', 'IS_PARTIAL',
            'METHOD', 'WEIGHTING', 'ELEMENT_WEIGHTS', 'MERGED_PARTIAL', 'MERGED_PARTIAL_AUTO',
            # Declustering weights (not a grade)
            'DECLUST_WEIGHT', 'WEIGHT'
        }
        
        # TRF-012: Filter out already-transformed columns from selection
        # (use the utility function from transform.py)
        valid = [c for c in nums if c.upper() not in ignore]
        valid = filter_transformed_columns(valid)
        
        self.col_combo.addItems(valid)

    def _on_col_changed(self, txt):
        """Handle column selection change."""
        self.trans_combo.setCurrentIndex(0)  # Reset to None
        
        # TRF-008: Update transform status display
        if txt in self.transformations:
            trans_info = self.transformations[txt]
            self.transform_status_lbl.setText(
                f"Already transformed: {trans_info['method']} → {trans_info['new_col']}"
            )
            self.transform_status_lbl.setStyleSheet("color: #ff9800; font-style: italic;")
        else:
            self.transform_status_lbl.setText("Not yet transformed")
            self.transform_status_lbl.setStyleSheet("color: #4caf50; font-style: italic;")

    def _on_method_changed(self, txt):
        """Handle transformation method selection change."""
        is_boxcox = "Box-Cox" in txt
        self.lambda_lbl.setVisible(is_boxcox)
        self.lambda_spin.setVisible(is_boxcox)
        
        is_ns = "Normal Score" in txt
        # Normal Score handles its own zero/distribution logic via class
        self.shift_chk.setEnabled(not is_ns)
        self.shift_spin.setEnabled(not is_ns)
        
        # TRF-002: Only enable weights for Normal Score
        self.use_weights_chk.setVisible(is_ns)

    # =========================================================
    # CORE MATH
    # =========================================================

    def _get_declustering_weights(self) -> Optional[np.ndarray]:
        """Get declustering weights if available and enabled."""
        if not self.use_weights_chk.isChecked():
            return None
        
        if self.original_data is None:
            return None
        
        # Look for weight column
        weight_cols = [c for c in self.original_data.columns if 'WEIGHT' in c.upper() and 'DECLUST' in c.upper()]
        if not weight_cols:
            return None
        
        return self.original_data[weight_cols[0]].values

    def _calculate_transform(self, col_name: str, method: str) -> tuple:
        """
        Calculate transformation and return (transformed_array, metadata).
        
        TRF-002: Supports declustering weights for Normal Score.
        TRF-009: Logs warnings for value replacements.
        """
        if self.original_data is None:
            return None, None
        
        raw = self.original_data[col_name].values.astype(float)
        mask = ~np.isnan(raw)
        valid = raw[mask]
        
        # Track value replacements for TRF-009
        n_zeros = 0
        n_negative = 0
        
        # Apply Shift
        shift_applied = False
        shift_value = 0.0
        if self.shift_chk.isChecked() and "Normal" not in method:
            shift_value = self.shift_spin.value()
            valid = valid + shift_value
            shift_applied = True
            
        result = np.full_like(raw, np.nan)
        meta: Dict[str, Any] = {}
        
        try:
            if "Log10" in method:
                # TRF-009: Count and warn about value replacements
                n_zeros = np.sum(valid == 0)
                n_negative = np.sum(valid < 0)
                if n_zeros > 0 or n_negative > 0:
                    logger.warning(
                        f"TRF-009: Log10 transform on '{col_name}': "
                        f"{n_zeros} zeros and {n_negative} negative values replaced with 1e-9"
                    )
                result[mask] = np.log10(np.maximum(valid, 1e-9))
                meta['value_replacements'] = {'zeros': n_zeros, 'negative': n_negative}
                
            elif "Log" in method:
                # TRF-009: Count and warn about value replacements
                n_zeros = np.sum(valid == 0)
                n_negative = np.sum(valid < 0)
                if n_zeros > 0 or n_negative > 0:
                    logger.warning(
                        f"TRF-009: Log transform on '{col_name}': "
                        f"{n_zeros} zeros and {n_negative} negative values replaced with 1e-9"
                    )
                result[mask] = np.log(np.maximum(valid, 1e-9))
                meta['value_replacements'] = {'zeros': n_zeros, 'negative': n_negative}
                
            elif "Square" in method:
                # TRF-009: Count and warn about value replacements
                n_negative = np.sum(valid < 0)
                if n_negative > 0:
                    logger.warning(
                        f"TRF-009: Square Root transform on '{col_name}': "
                        f"{n_negative} negative values replaced with 0"
                    )
                result[mask] = np.sqrt(np.maximum(valid, 0))
                meta['value_replacements'] = {'negative': n_negative}
                
            elif "Box-Cox" in method and SCIPY_AVAILABLE:
                # Auto-fit lambda if 0, else use spinner
                l_val = self.lambda_spin.value()
                clean = valid[valid > 0]
                
                # TRF-009: Warn about excluded values
                n_excluded = np.sum(valid <= 0)
                if n_excluded > 0:
                    logger.warning(
                        f"TRF-009: Box-Cox transform on '{col_name}': "
                        f"{n_excluded} non-positive values excluded (Box-Cox requires > 0)"
                    )
                
                if len(clean) > 0:
                    if l_val == 0:
                        trans, l_opt = stats.boxcox(clean)
                        self.lambda_spin.setValue(l_opt)  # Update UI
                        l_val = l_opt
                    else:
                        trans = stats.boxcox(clean, lmbda=l_val)
                    
                    # Map back to full array
                    # Note: Box-Cox strictly requires > 0
                    pos_mask = (raw > 0) & mask
                    result[:] = np.nan
                    if shift_applied:
                        result[pos_mask] = stats.boxcox(raw[pos_mask] + shift_value, lmbda=l_val)
                    else:
                        result[pos_mask] = stats.boxcox(raw[pos_mask], lmbda=l_val)
                        
                meta['boxcox_lambda'] = l_val
                meta['value_replacements'] = {'non_positive_excluded': n_excluded}
                        
            elif "Normal Score" in method:
                # USE THE ROBUST TRANSFORMER CLASS with TRF-002 weights support
                nst = NormalScoreTransformer()
                
                # Get declustering weights if enabled
                weights = self._get_declustering_weights()
                if weights is not None:
                    valid_weights = weights[mask]
                    nst.fit(valid, weights=valid_weights)
                    meta['weights_used'] = True
                    logger.info(f"TRF-002: Normal Score transform using declustering weights")
                else:
                    nst.fit(valid)
                    meta['weights_used'] = False
                
                trans = nst.transform(valid)
                result[mask] = trans
                meta['transformer_object'] = nst  # Store for registry
                meta['transformer_provenance'] = nst.get_provenance()
                
        except Exception as e:
            logger.error(f"Math error in transformation: {e}", exc_info=True)
            return None, None
            
        return result, meta

    # =========================================================
    # ACTIONS
    # =========================================================

    def _preview(self):
        """Preview the transformation without applying."""
        col = self.col_combo.currentText()
        method = self.trans_combo.currentText()
        if not col:
            return

        trans_vals, _ = self._calculate_transform(col, method)
        if trans_vals is None:
            return

        # Store for export
        self._preview_col = col
        self._preview_method = method
        self._preview_trans_vals = trans_vals

        # Update Plot
        if MATPLOTLIB_AVAILABLE:
            self.canvas.figure.clear()
            ax1 = self.canvas.figure.add_subplot(121)
            ax2 = self.canvas.figure.add_subplot(122)

            # Styles
            for ax in [ax1, ax2]:
                ax.set_facecolor('#2b2b2b')
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.grid(alpha=0.3, color='#555')

            orig = self.original_data[col].values
            orig = orig[~np.isnan(orig)]

            ax1.hist(orig, bins=30, color='skyblue', alpha=0.7, edgecolor='#1e88e5')
            ax1.set_title("Original", color='white', fontweight='bold')
            ax1.set_xlabel(col, color='white')
            ax1.set_ylabel("Frequency", color='white')

            clean_trans = trans_vals[~np.isnan(trans_vals)]
            ax2.hist(clean_trans, bins=30, color='lightgreen', alpha=0.7, edgecolor='#43a047')
            ax2.set_title(f"Transformed ({method})", color='white', fontweight='bold')
            ax2.set_xlabel(f"{col} ({method})", color='white')
            ax2.set_ylabel("Frequency", color='white')

            if "Normal" in method and SCIPY_AVAILABLE:
                # Overlay Gaussian
                x = np.linspace(-4, 4, 100)
                ax2.plot(x, stats.norm.pdf(x) * len(clean_trans) * (x[1] - x[0]), 'r--',
                         linewidth=2, label='Standard Normal')
                ax2.legend(facecolor='#333', edgecolor='#555', labelcolor='white')

            self.canvas.figure.tight_layout()
            self.canvas.draw()

            # Enable chart export button
            if self.chart_export_btn:
                self.chart_export_btn.setEnabled(True)

        # Update Data Preview Table
        self._update_data_preview_table(col, method, trans_vals)

    def _update_data_preview_table(self, col_name: str, method: str, trans_vals: np.ndarray):
        """Update the data preview table with full transformed values and summary statistics."""
        if self.original_data is None or trans_vals is None:
            self.tab_data.setRowCount(0)
            self.tab_data.setColumnCount(0)
            self.data_stats_lbl.setText("No data loaded")
            self.data_export_btn.setEnabled(False)
            return

        orig_series = self.original_data[col_name]
        valid_mask = ~np.isnan(orig_series.values) & ~np.isnan(trans_vals)

        if not np.any(valid_mask):
            self.tab_data.setRowCount(0)
            self.tab_data.setColumnCount(0)
            self.data_stats_lbl.setText("No valid data")
            self.data_export_btn.setEnabled(False)
            return

        # Get ALL valid indices (full data preview)
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        n_total = len(orig_series)

        # Update stats label
        self.data_stats_lbl.setText(f"Showing {n_valid:,} valid samples of {n_total:,} total rows")

        # Calculate summary statistics
        orig_valid = orig_series.values[valid_mask]
        trans_valid = trans_vals[valid_mask]

        orig_stats = (
            f"Original ({col_name}):\n"
            f"  Mean: {np.mean(orig_valid):.4f}\n"
            f"  Std Dev: {np.std(orig_valid):.4f}\n"
            f"  Min: {np.min(orig_valid):.4f}\n"
            f"  Max: {np.max(orig_valid):.4f}\n"
            f"  Median: {np.median(orig_valid):.4f}"
        )
        trans_stats = (
            f"Transformed ({method}):\n"
            f"  Mean: {np.mean(trans_valid):.4f}\n"
            f"  Std Dev: {np.std(trans_valid):.4f}\n"
            f"  Min: {np.min(trans_valid):.4f}\n"
            f"  Max: {np.max(trans_valid):.4f}\n"
            f"  Median: {np.median(trans_valid):.4f}"
        )

        # Add skewness/kurtosis if scipy available
        if SCIPY_AVAILABLE:
            orig_stats += f"\n  Skewness: {stats.skew(orig_valid):.4f}"
            orig_stats += f"\n  Kurtosis: {stats.kurtosis(orig_valid):.4f}"
            trans_stats += f"\n  Skewness: {stats.skew(trans_valid):.4f}"
            trans_stats += f"\n  Kurtosis: {stats.kurtosis(trans_valid):.4f}"

        self.orig_stats_lbl.setText(orig_stats)
        self.trans_stats_lbl.setText(trans_stats)

        # Set up table with full data
        self.tab_data.setRowCount(n_valid)
        self.tab_data.setColumnCount(3)
        self.tab_data.setHorizontalHeaderLabels(["Row", f"Original ({col_name})", f"Transformed ({method})"])
        self.tab_data.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tab_data.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tab_data.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        # Populate table with all valid data
        for i, idx in enumerate(valid_indices):
            # Row number (1-indexed)
            self.tab_data.setItem(i, 0, QTableWidgetItem(str(idx + 1)))

            # Original value
            orig_val = orig_series.iloc[idx]
            self.tab_data.setItem(i, 1, QTableWidgetItem(f"{orig_val:.6f}"))

            # Transformed value
            trans_val = trans_vals[idx]
            self.tab_data.setItem(i, 2, QTableWidgetItem(f"{trans_val:.6f}"))

        # Enable data export button
        self.data_export_btn.setEnabled(True)

    def _apply(self):
        """Apply the transformation and create new column."""
        col = self.col_combo.currentText()
        method = self.trans_combo.currentText()
        
        if not col or method == "None":
            QMessageBox.warning(self, "Invalid Selection", "Please select a column and transformation method.")
            return
        
        # TRF-008: Check if column already transformed
        if col in self.transformations:
            reply = QMessageBox.question(
                self,
                "Column Already Transformed",
                f"Column '{col}' has already been transformed using {self.transformations[col]['method']}.\n\n"
                f"Applying another transformation may invalidate the back-transform.\n\n"
                "Do you want to proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            
            logger.warning(f"TRF-008: User applying second transformation to '{col}' (stacked transform)")
        
        vals, meta = self._calculate_transform(col, method)
        if vals is None:
            QMessageBox.critical(self, "Error", "Transformation failed. Check the log for details.")
            return
        
        # Name new column
        suffix = {"Log": "_LN", "Log10": "_LOG", "Box-Cox": "_BC", "Normal Score": "_NS", "Square Root": "_SQRT"}
        new_name = f"{col}{suffix.get(method, '_TR')}"
        
        self.transformed_data[new_name] = vals
        
        # =====================================================================
        # TRF-004 COMPLIANCE: Store full provenance metadata
        # =====================================================================
        source_type = "composites" if "Composited" in self.data_source_combo.currentText() else "assays"
        
        self.transformations[col] = {
            "method": method,
            "new_col": new_name,
            "transformer": meta.get('transformer_object'),  # Critical for Back-Transform
            # TRF-004: Provenance metadata
            "parent_data_key": source_type,
            "source_was_raw_assays": source_type == "assays",
            "transformation_timestamp": datetime.now().isoformat(),
            "sample_count": len(self.original_data) if self.original_data is not None else 0,
            "weights_used": meta.get('weights_used', False),
            "value_replacements": meta.get('value_replacements', {}),
            "transformer_provenance": meta.get('transformer_provenance', {}),
        }
        
        if meta.get('transformer_object'):
            self.active_transformers[col] = meta['transformer_object']
        
        # TRF-008: Add to transform history
        self._transform_history.append({
            "column": col,
            "method": method,
            "new_col": new_name,
            "source": source_type,
            "timestamp": datetime.now().isoformat()
        })
        self._update_history_table()
        
        # Update column status
        self._on_col_changed(col)
            
        QMessageBox.information(self, "Applied", f"Created column: {new_name}")

    def _update_history_table(self):
        """Update the transform history table."""
        self.history_table.setRowCount(len(self._transform_history))
        for i, entry in enumerate(self._transform_history):
            self.history_table.setItem(i, 0, QTableWidgetItem(entry.get('column', '')))
            self.history_table.setItem(i, 1, QTableWidgetItem(entry.get('method', '')))
            self.history_table.setItem(i, 2, QTableWidgetItem(entry.get('new_col', '')))
            self.history_table.setItem(i, 3, QTableWidgetItem(entry.get('source', '')))

    def _send_to_registry(self):
        """Update the Global Registry so other panels see the new columns."""
        if not self.registry:
            QMessageBox.warning(self, "Registry Not Available", 
                              "DataRegistry is not connected. Cannot update registry.")
            return
        if self.transformed_data is None:
            QMessageBox.warning(self, "No Data", 
                              "No transformed data available. Please apply a transformation first.")
            return
        
        # We need to preserve the structure of the registry data
        # We only update the dataframe that was edited
        if self.registry_snapshot is None:
            # Try to get current registry data
            existing = self.registry.get_drillhole_data()
            if existing:
                self.registry_snapshot = existing
            else:
                QMessageBox.warning(self, "No Registry Data", 
                                  "No data found in registry. Please load drillhole data first.")
                return
        
        new_data = self.registry_snapshot.copy()
        
        source_type = "composites" if "Composited" in self.data_source_combo.currentText() else "assays"
        if source_type == "composites":
            new_data['composites'] = self.transformed_data
        else:
            new_data['assays'] = self.transformed_data
        
        # TRF-004: Include provenance in registry metadata
        transformation_metadata = {
            "transformations": {k: {
                "method": v.get("method"),
                "new_col": v.get("new_col"),
                "parent_data_key": v.get("parent_data_key"),
                "source_was_raw_assays": v.get("source_was_raw_assays"),
                "transformation_timestamp": v.get("transformation_timestamp"),
                "weights_used": v.get("weights_used", False),
            } for k, v in self.transformations.items()},
            "transform_history": self._transform_history,
        }
            
        # Register Updated Data
        self.registry.register_drillhole_data(
            new_data, 
            source_panel="Grade Transformer",
            metadata=transformation_metadata
        )
        
        # Register Transformer Objects (for SGSIM back-transform)
        # We pass the dictionary of {col_name: NormalScoreTransformer}
        if self.active_transformers:
            self.registry.register_transformers(self.active_transformers)
        
        # Also register transformation metadata separately for downstream access
        if hasattr(self.registry, 'register_transformation_metadata'):
            self.registry.register_transformation_metadata(
                transformation_metadata,
                source_panel="GradeTransformationPanel"
            )
            
        QMessageBox.information(
            self,
            "Success",
            f"Data updated in Registry.\n"
            f"Transformers registered for SGSIM.\n"
            f"Provenance metadata stored for audit trail."
        )

    # =========================================================
    # EXPORT FUNCTIONALITY
    # =========================================================

    def _export_chart(self):
        """Export the distribution chart as PNG, SVG, or PDF."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, 'canvas'):
            QMessageBox.warning(self, "Export Error", "No chart available to export.")
            return

        # Get filename from user
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Chart",
            f"transformation_chart_{self._preview_col}_{self._preview_method}",
            "PNG Image (*.png);;SVG Vector (*.svg);;PDF Document (*.pdf)"
        )

        if not filename:
            return

        try:
            fig = self.canvas.figure

            # Determine format from filter or extension
            if filename.endswith('.svg') or 'SVG' in selected_filter:
                if not filename.endswith('.svg'):
                    filename += '.svg'
                fig.savefig(filename, format='svg', dpi=300, bbox_inches='tight',
                            facecolor='#2b2b2b', edgecolor='none')
            elif filename.endswith('.pdf') or 'PDF' in selected_filter:
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                fig.savefig(filename, format='pdf', dpi=300, bbox_inches='tight',
                            facecolor='#2b2b2b', edgecolor='none')
            else:
                if not filename.endswith('.png'):
                    filename += '.png'
                fig.savefig(filename, format='png', dpi=300, bbox_inches='tight',
                            facecolor='#2b2b2b', edgecolor='none')

            QMessageBox.information(self, "Exported", f"Chart exported successfully:\n{filename}")
            logger.info(f"Chart exported to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export chart:\n{str(e)}")
            logger.error(f"Chart export error: {e}", exc_info=True)

    def _export_data(self):
        """Export the transformed data preview to CSV."""
        if self.original_data is None or not hasattr(self, '_preview_trans_vals'):
            QMessageBox.warning(self, "Export Error", "No data available to export. Run Preview first.")
            return

        col_name = getattr(self, '_preview_col', 'grade')
        method = getattr(self, '_preview_method', 'transformed')
        trans_vals = self._preview_trans_vals

        # Get filename from user
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transformed Data",
            f"transformation_data_{col_name}_{method}",
            "CSV File (*.csv);;Excel File (*.xlsx)"
        )

        if not filename:
            return

        try:
            # Build export DataFrame
            orig_series = self.original_data[col_name]
            valid_mask = ~np.isnan(orig_series.values) & ~np.isnan(trans_vals)

            export_df = pd.DataFrame({
                'Row': np.arange(1, len(orig_series) + 1),
                f'Original_{col_name}': orig_series.values,
                f'Transformed_{method}': trans_vals,
                'Valid': valid_mask
            })

            # Add summary statistics row at the end
            stats_row = {
                'Row': 'STATS',
                f'Original_{col_name}': '',
                f'Transformed_{method}': '',
                'Valid': ''
            }

            # Filter to valid data only for the main export, add full data with validity flag
            if filename.endswith('.xlsx'):
                if not filename.endswith('.xlsx'):
                    filename += '.xlsx'

                # Excel export with multiple sheets
                with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                    # Full data sheet
                    export_df.to_excel(writer, sheet_name='Full_Data', index=False)

                    # Valid data only sheet
                    valid_df = export_df[export_df['Valid']].drop(columns=['Valid'])
                    valid_df.to_excel(writer, sheet_name='Valid_Data', index=False)

                    # Summary statistics sheet
                    orig_valid = orig_series.values[valid_mask]
                    trans_valid = trans_vals[valid_mask]

                    stats_data = {
                        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Q1', 'Q3'],
                        f'Original_{col_name}': [
                            len(orig_valid),
                            np.mean(orig_valid),
                            np.std(orig_valid),
                            np.min(orig_valid),
                            np.max(orig_valid),
                            np.median(orig_valid),
                            np.percentile(orig_valid, 25),
                            np.percentile(orig_valid, 75)
                        ],
                        f'Transformed_{method}': [
                            len(trans_valid),
                            np.mean(trans_valid),
                            np.std(trans_valid),
                            np.min(trans_valid),
                            np.max(trans_valid),
                            np.median(trans_valid),
                            np.percentile(trans_valid, 25),
                            np.percentile(trans_valid, 75)
                        ]
                    }

                    if SCIPY_AVAILABLE:
                        stats_data['Statistic'].extend(['Skewness', 'Kurtosis'])
                        stats_data[f'Original_{col_name}'].extend([
                            stats.skew(orig_valid), stats.kurtosis(orig_valid)
                        ])
                        stats_data[f'Transformed_{method}'].extend([
                            stats.skew(trans_valid), stats.kurtosis(trans_valid)
                        ])

                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Summary_Stats', index=False)

            else:
                # CSV export
                if not filename.endswith('.csv'):
                    filename += '.csv'

                # Export valid data only
                valid_df = export_df[export_df['Valid']].drop(columns=['Valid'])
                valid_df.to_csv(filename, index=False)

            QMessageBox.information(self, "Exported", f"Data exported successfully:\n{filename}")
            logger.info(f"Transformed data exported to {filename}")

        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export data:\n{str(e)}")
            logger.error(f"Data export error: {e}", exc_info=True)

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Data source and variable
            settings['data_source'] = get_safe_widget_value(self, 'data_source_combo')
            settings['variable'] = get_safe_widget_value(self, 'var_combo')
            settings['transform_type'] = get_safe_widget_value(self, 'transform_combo')
            
            # Transformation history (saved for audit purposes)
            if hasattr(self, '_transform_history') and self._transform_history:
                settings['transform_history'] = self._transform_history
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save grade transformation panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Data source and variable
            set_safe_widget_value(self, 'data_source_combo', settings.get('data_source'))
            set_safe_widget_value(self, 'var_combo', settings.get('variable'))
            set_safe_widget_value(self, 'transform_combo', settings.get('transform_type'))
                
            logger.info("Restored grade transformation panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore grade transformation panel settings: {e}")

    def refresh_theme(self):
        """Refresh styles when theme changes."""
        self.setStyleSheet(get_analysis_panel_stylesheet())