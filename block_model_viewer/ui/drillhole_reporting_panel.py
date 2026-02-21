"""
Drillhole Reporting Panel with Lineage Enforcement

This panel computes and exports drillhole statistics while enforcing 
STAT-001 through STAT-006 lineage constraints.

KEY FEATURES (STAT-001 through STAT-006):
- Prefers composites over raw assays (STAT-001)
- Applies declustering weights when available (STAT-002)
- Provides grade-tonnage tables with clear assumptions (STAT-008)
- Enforces validation status checks (STAT-005)
- Clear provenance in exports (JORC/SAMREC ready)

MODERN UI:
- Horizontal layout: controls left, data display right
- Dark theme with accent colors
- No Stop/Close buttons (uses _build_ui pattern)
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import logging
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QMessageBox, QLineEdit, QFrame,
    QSplitter
)
from PyQt6.QtCore import Qt

from .panel_manager import PanelCategory, DockArea
from .base_analysis_panel import BaseAnalysisPanel
from .modern_styles import get_analysis_panel_stylesheet, get_theme_colors, get_current_theme
# TRF-012: Import filter for transformed columns
from ..models.transform import filter_transformed_columns, is_transformed_column

logger = logging.getLogger(__name__)


class DrillholeReportingPanel(BaseAnalysisPanel):
    """
    Drillhole Reporting Panel - Statistics and Grade-Tonnage Analysis.
    
    LINEAGE ENFORCEMENT:
    - Prefers composited data over raw assays
    - Warns user when falling back to raw assays
    - Checks validation status before computing statistics
    - Integrates declustering weights when available
    
    TONNAGE ASSUMPTIONS:
    - If no density column, tonnage = interval length (metres)
    - For true tonnage, provide 'density' or 'DENSITY' column
    - Tonnage = length × density × cross-sectional area (assumed 1 m²)
    
    AUDIT TRAIL:
    - Data source type tracked in _data_source_type attribute
    - Validation status checked at each statistics computation
    - Declustering status tracked in _using_declustered attribute
    
    MODERN UI:
    - Uses _build_ui() to skip base class Stop/Close buttons
    - Horizontal layout with QSplitter (controls left, data right)
    - Dark theme styling
    """
    # PanelManager metadata
    PANEL_ID = "DrillholeReportingPanel"
    PANEL_NAME = "DrillholeReporting Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "drillhole_reporting"
    
    def __init__(self, parent=None):
        self._df: Optional[pd.DataFrame] = None
        self._declust_weights: Optional[pd.Series] = None
        self._data_source_type: str = "unknown"  # 'composites', 'assays', or 'unknown'
        self._using_declustered: bool = False
        self._validation_status: str = "NOT_RUN"

        # Track if data has changed since last view (must init BEFORE registry connection)
        self._pending_data_update: bool = False
        self._ui_ready: bool = False
        self._registry_data = None  # Store raw registry data

        super().__init__(parent, panel_id="drillhole_reporting")

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
        self.setWindowTitle("Drillhole Reporting")

        # CRITICAL: Call _build_ui() after super().__init__() since BaseAnalysisPanel
        # detected we have _build_ui() and skipped setup_ui()
        self._build_ui()

        # Mark UI as ready
        self._ui_ready = True
        
        # Connect Registry
        reg = self.get_registry()
        if reg:
            reg.drillholeDataLoaded.connect(self._on_data_loaded)
            # Also listen for declustering results
            if hasattr(reg, 'declusteringResultsLoaded'):
                reg.declusteringResultsLoaded.connect(self._on_declustering_loaded)
            d = reg.get_drillhole_data()
            if d is not None:
                self._on_data_loaded(d)
            # Check for existing declustering results
            self._load_declustering_weights()
    
    def _get_primary_btn_style(self) -> str:
        """Primary button style with accent color."""
        colors = get_theme_colors()
        return f"""
            QPushButton {{
                background-color: {colors.ACCENT_PRIMARY};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {colors.ACCENT_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {colors.ACCENT_PRESSED};
            }}
            QPushButton:disabled {{
                background-color: {colors.BORDER};
                color: {colors.TEXT_DISABLED};
            }}
        """
    
    def _get_secondary_btn_style(self) -> str:
        """Secondary button style."""
        colors = get_theme_colors()
        return f"""
            QPushButton {{
                background-color: {colors.ELEVATED_BG};
                color: {colors.TEXT_PRIMARY};
                border: 1px solid {colors.BORDER};
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
                border-color: {colors.ACCENT_PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {colors.BORDER};
            }}
        """

    def _build_ui(self):
        """Build modern horizontal layout UI. Using _build_ui skips base class Stop/Close buttons."""
        # Apply modern stylesheet
        self.setStyleSheet(get_analysis_panel_stylesheet())
        colors = get_theme_colors()

        # Use inherited main_layout from BaseAnalysisPanel
        layout = self.main_layout
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # =========================================================
        # LINEAGE BANNER (STAT-001, STAT-005)
        # =========================================================
        self.lineage_banner = QFrame()
        self.lineage_banner.setStyleSheet(f"background-color: {colors.ELEVATED_BG}; border: 1px solid {colors.BORDER}; border-radius: 4px;")
        banner_layout = QHBoxLayout(self.lineage_banner)
        banner_layout.setContentsMargins(10, 6, 10, 6)
        self.lineage_label = QLabel("⏳ No data loaded")
        self.lineage_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 11px;")
        banner_layout.addWidget(self.lineage_label)
        banner_layout.addStretch()
        layout.addWidget(self.lineage_banner)

        # New data notification banner (hidden by default)
        self._new_data_banner = QLabel("🔔 New data available! Click 'Refresh' to load the latest drillhole data.")
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

        # =========================================================
        # HORIZONTAL SPLITTER: Controls Left | Data Right
        # =========================================================
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {colors.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {colors.ACCENT_PRIMARY};
            }}
        """)
        
        # =========================================================
        # LEFT PANEL: Controls
        # =========================================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(0, 0, 8, 0)
        
        # Helper for groupbox styling - rely on get_analysis_panel_stylesheet() for base styles
        groupbox_style = f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                color: {colors.TEXT_PRIMARY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {colors.ACCENT_PRIMARY};
            }}
        """

        # Data Loading Group
        data_group = QGroupBox("Data Loading")
        data_group.setStyleSheet(groupbox_style)
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(8)

        self.refresh_btn = QPushButton("🔄 Refresh from Registry")
        self.refresh_btn.setToolTip("Refresh data from registry (get latest composites/assays)")
        self.refresh_btn.clicked.connect(self._manual_refresh)
        self.refresh_btn.setStyleSheet(self._get_primary_btn_style())
        data_layout.addWidget(self.refresh_btn)

        left_layout.addWidget(data_group)

        # Element Selection Group
        elem_group = QGroupBox("Element Selection")
        elem_group.setStyleSheet(groupbox_style)
        elem_layout = QVBoxLayout(elem_group)
        elem_layout.setSpacing(8)
        
        elem_lbl = QLabel("Grade Column:")
        elem_lbl.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-weight: normal;")
        elem_layout.addWidget(elem_lbl)
        
        self.col_combo = QComboBox()
        self.col_combo.setMinimumWidth(150)
        self.col_combo.currentTextChanged.connect(self._run_stats)
        # Rely on stylesheet from get_analysis_panel_stylesheet() for combo
        elem_layout.addWidget(self.col_combo)
        left_layout.addWidget(elem_group)
        
        # Cutoff Configuration Group
        cutoff_group = QGroupBox("Cutoff Grades")
        cutoff_group.setStyleSheet(groupbox_style)
        cutoff_layout = QVBoxLayout(cutoff_group)
        cutoff_layout.setSpacing(8)
        
        cutoff_lbl = QLabel("Values (comma-separated):")
        cutoff_lbl.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-weight: normal;")
        cutoff_layout.addWidget(cutoff_lbl)
        
        self.cutoffs_edit = QLineEdit("0, 0.5, 1.0, 2.0, 5.0")
        self.cutoffs_edit.setPlaceholderText("e.g., 0, 0.5, 1.0, 2.0, 5.0")
        # Rely on stylesheet for line edit
        cutoff_layout.addWidget(self.cutoffs_edit)
        
        self.btn_suggest = QPushButton("Auto-Suggest")
        self.btn_suggest.setToolTip("Suggest cutoffs based on data percentiles")
        self.btn_suggest.clicked.connect(self._auto_suggest_cutoffs)
        self.btn_suggest.setStyleSheet(self._get_secondary_btn_style())
        cutoff_layout.addWidget(self.btn_suggest)
        
        left_layout.addWidget(cutoff_group)
        
        # Action Buttons
        actions_group = QGroupBox("Actions")
        actions_group.setStyleSheet(groupbox_style)
        actions_layout = QVBoxLayout(actions_group)
        actions_layout.setSpacing(8)
        
        self.btn_run = QPushButton("Compute Statistics")
        self.btn_run.clicked.connect(self._run_stats)
        self.btn_run.setStyleSheet(self._get_primary_btn_style())
        actions_layout.addWidget(self.btn_run)
        
        self.btn_export = QPushButton("Export Report")
        self.btn_export.clicked.connect(self._export)
        self.btn_export.setStyleSheet(self._get_secondary_btn_style())
        actions_layout.addWidget(self.btn_export)
        
        left_layout.addWidget(actions_group)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 10px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # =========================================================
        # RIGHT PANEL: Data Display
        # =========================================================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(8, 0, 0, 0)
        
        # Statistics Output
        stats_group = QGroupBox("Summary Statistics")
        stats_group.setStyleSheet(groupbox_style)
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {colors.ELEVATED_BG};
                color: {colors.TEXT_PRIMARY};
                border: 1px solid {colors.BORDER};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }}
        """)
        stats_layout.addWidget(self.stats_text)
        right_layout.addWidget(stats_group)
        
        # Grade-Tonnage Table
        gt_group = QGroupBox("Grade-Tonnage Table")
        gt_group.setStyleSheet(groupbox_style)
        gt_layout = QVBoxLayout(gt_group)
        
        self.gt_table = QTableWidget()
        self.gt_table.setColumnCount(5)
        self.gt_table.setHorizontalHeaderLabels(["Cutoff", "Tonnage*", "Avg Grade", "Metal", "Wtd"])
        self.gt_table.setToolTip(
            "* Tonnage assumes 1 metre interval = 1 tonne (no density).\n"
            "For accurate tonnage, provide density column in source data.\n"
            "'Wtd' column shows ✓ if declustering weights were applied."
        )
        # Rely on get_analysis_panel_stylesheet() for table styling
        self.gt_table.horizontalHeader().setStretchLastSection(True)
        self.gt_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        gt_layout.addWidget(self.gt_table)
        right_layout.addWidget(gt_group)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 500])  # Controls narrower, data wider
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        
        layout.addWidget(splitter, 1)
        
        # Defensive: Remove any Stop/Close buttons that might have been added by base class
        if hasattr(self, 'stop_button') and self.stop_button:
            self.stop_button.setParent(None)
            self.stop_button.deleteLater()
            self.stop_button = None
        if hasattr(self, 'close_button') and self.close_button:
            self.close_button.setParent(None)
            self.close_button.deleteLater()
            self.close_button = None

    def _on_data_loaded(self, data):
        """
        Handle drillhole data loaded from registry.

        Shows notification banner if panel is visible, otherwise marks for refresh.

        LINEAGE ENFORCEMENT (STAT-001):
        - Explicitly tracks data source type
        - Warns user when falling back to raw assays
        - Updates lineage banner with data source status
        """
        # Store the data for later processing
        self._registry_data = data

        # If panel is visible, show notification banner (user decides when to refresh)
        # If panel is hidden, auto-update on next show
        if self.isVisible() and self._ui_ready:
            # Show notification banner - let user decide when to refresh
            if hasattr(self, '_new_data_banner'):
                self._new_data_banner.setVisible(True)
            self._pending_data_update = True
        else:
            # Panel not visible - auto-update when shown
            self._pending_data_update = True
            # If UI is ready but panel just hasn't been shown yet, apply now
            if self._ui_ready:
                self._apply_data_update()

    def _apply_data_update(self):
        """Apply the pending data update from registry to UI."""
        # Only apply if UI is ready
        if not getattr(self, '_ui_ready', False):
            return

        # Get stored registry data
        data = getattr(self, '_registry_data', None)
        if data is None:
            logger.debug("DrillholeReportingPanel: No registry data to apply")
            return

        df = None
        self._data_source_type = "unknown"

        # LINEAGE: Check validation status first (STAT-005)
        reg = self.get_registry()
        if reg:
            self._validation_status = reg.get_drillholes_validation_status()

        # LINEAGE: Prefer composites over raw assays
        composites = data.get('composites')
        assays = data.get('assays')
        
        if isinstance(composites, pd.DataFrame) and not composites.empty:
            df = composites
            self._data_source_type = "composites"
            logger.info(f"LINEAGE: DrillholeReporting using COMPOSITES ({len(df)} samples)")
        elif isinstance(assays, pd.DataFrame) and not assays.empty:
            df = assays
            self._data_source_type = "assays"
            # STAT-001: Explicit warning when falling back to raw assays
            logger.warning(
                f"LINEAGE WARNING: DrillholeReporting using RAW ASSAYS ({len(df)} samples). "
                "Statistics may be biased due to inconsistent sample support. "
                "Consider running compositing first for JORC/SAMREC compliance."
            )

        if df is not None and not df.empty:
            self._df = df

            # Update lineage banner (STAT-001, STAT-005)
            self._update_lineage_banner()

            # Check for declustering weights
            self._load_declustering_weights()

            # Populate Columns
            nums = df.select_dtypes(include=[np.number]).columns
            ignore = {
                'from','to','depth_from','depth_to','x','y','z','holeid','hole_id','global_interval_id',
                # Compositing metadata columns
                'sample_count','total_mass','total_length','support','is_partial',
                'method','weighting','element_weights','merged_partial','merged_partial_auto',
                # Declustering columns
                'declust_weight', 'declust_cell'
            }
            cols = [c for c in nums if c.lower() not in ignore]

            # =====================================================================
            # TRF-012 COMPLIANCE: Filter out transformed columns
            # =====================================================================
            # Prevents statistical leakage of transformed values into grade-tonnage
            # reports. Users must use physical (untransformed) grades for reporting.
            physical_cols = filter_transformed_columns(cols)

            # Log if any transformed columns were filtered
            n_filtered = len(cols) - len(physical_cols)
            if n_filtered > 0:
                logger.info(
                    f"TRF-012: Filtered {n_filtered} transformed column(s) from grade selection. "
                    f"Reporting uses physical grades only to prevent statistical leakage."
                )

            self.col_combo.blockSignals(True)
            self.col_combo.clear()
            self.col_combo.addItems(sorted(physical_cols))
            self.col_combo.blockSignals(False)

            if physical_cols:
                self._run_stats()
                self._update_status(f"✅ Loaded {len(df):,} samples, {len(physical_cols)} grade columns", "green")
            else:
                self._update_status("⚠️ No grade columns found in data", "orange")
        else:
            self._update_lineage_banner()
            self._update_status("⏳ No data loaded", "gray")

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
                    self._registry_data = data
                    self._apply_data_update()
                    logger.info("DrillholeReportingPanel: Manual refresh completed")
                else:
                    QMessageBox.information(
                        self,
                        "No Data",
                        "No drillhole data available in registry.\n\n"
                        "Please load drillhole data first using the Drillholes menu."
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Registry Error",
                    "DataRegistry not available. Cannot refresh."
                )
        except Exception as e:
            logger.error(f"Failed to refresh data: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Refresh Error",
                f"Failed to refresh data from registry:\n{e}"
            )

    def showEvent(self, event):
        """Auto-refresh when panel becomes visible."""
        super().showEvent(event)

        # If there's a pending update, apply it now
        if getattr(self, '_pending_data_update', False):
            self._apply_data_update()
    
    def _update_status(self, text: str, color: str = "gray"):
        """Update status label with colored text."""
        colors = get_theme_colors()
        color_map = {
            "green": colors.SUCCESS,
            "orange": colors.WARNING,
            "red": colors.ERROR,
            "gray": colors.TEXT_SECONDARY
        }
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color_map.get(color, color)}; font-size: 10px;")
    
    def _update_lineage_banner(self):
        """Update the lineage status banner with current data source info."""
        colors = get_theme_colors()
        is_dark = get_current_theme() == "dark"
        
        if self._df is None:
            self.lineage_label.setText("⏳ No data loaded")
            self.lineage_banner.setStyleSheet(f"background-color: {colors.ELEVATED_BG}; border: 1px solid {colors.BORDER}; border-radius: 4px;")
            return
        
        parts = []
        
        # Data source status - use semantic colors with theme-appropriate backgrounds
        if self._data_source_type == "composites":
            parts.append("✅ Composites")
            bg_color = "#1b3a1b" if is_dark else "#e8f5e9"  # Dark green tint / Light green tint
            border_color = colors.SUCCESS
        elif self._data_source_type == "assays":
            parts.append("⚠️ Raw Assays (not composited)")
            bg_color = "#3a3a1b" if is_dark else "#fff3e0"  # Dark yellow tint / Light orange tint
            border_color = colors.WARNING
        else:
            parts.append("❓ Unknown source")
            bg_color = colors.ELEVATED_BG
            border_color = colors.BORDER
        
        # Validation status
        if self._validation_status == "PASS":
            parts.append("✅ Validated")
        elif self._validation_status == "WARN":
            parts.append("⚠️ Validated (warnings)")
        elif self._validation_status == "FAIL":
            parts.append("⚠️ Validation issues")
            # Don't block, just warn
        else:
            parts.append("❓ Not validated")
        
        # Declustering status
        if self._using_declustered:
            parts.append("✅ Declustered")
        else:
            parts.append("⚪ No declustering")
        
        # Sample count
        parts.append(f"N={len(self._df):,}")
        
        self.lineage_label.setText(" | ".join(parts))
        self.lineage_banner.setStyleSheet(
            f"background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 4px;"
        )

    def _auto_suggest_cutoffs(self):
        """Auto-suggest cutoff values based on data percentiles.

        Uses 0, P25, P50, P75, P95 percentiles to provide meaningful cutoffs
        that span the grade distribution. Works for any commodity.
        """
        if self._df is None or self._df.empty:
            QMessageBox.warning(self, "No Data", "Load data first to auto-suggest cutoffs.")
            return

        col = self.col_combo.currentText()
        if not col or col not in self._df.columns:
            QMessageBox.warning(self, "No Column", "Select an element column first.")
            return

        try:
            values = self._df[col].dropna()
            if len(values) == 0:
                QMessageBox.warning(self, "No Data", f"No valid data for '{col}'.")
                return

            # Calculate percentiles
            p0 = 0  # Always include 0 as first cutoff
            p25 = values.quantile(0.25)
            p50 = values.quantile(0.50)
            p75 = values.quantile(0.75)
            p95 = values.quantile(0.95)

            # Determine decimal places based on data magnitude
            grade_max = values.max()
            if grade_max > 50:
                decimals = 1
            elif grade_max > 10:
                decimals = 1
            elif grade_max > 1:
                decimals = 2
            else:
                decimals = 3

            # Round percentiles
            cutoffs = [p0, round(p25, decimals), round(p50, decimals), round(p75, decimals), round(p95, decimals)]
            # Remove duplicates while preserving order
            unique_cutoffs = []
            for c in cutoffs:
                if c not in unique_cutoffs:
                    unique_cutoffs.append(c)

            # Format as comma-separated string
            cutoff_str = ", ".join(str(c) for c in unique_cutoffs)
            self.cutoffs_edit.setText(cutoff_str)

            logger.info(f"DrillholeReporting: Auto-suggested cutoffs for '{col}': {cutoff_str}")
            self._update_status(f"Suggested cutoffs: {cutoff_str}", "green")

        except Exception as e:
            logger.warning(f"DrillholeReporting: Could not auto-suggest cutoffs: {e}")
            QMessageBox.warning(self, "Error", f"Could not calculate cutoffs: {e}")

    def _load_declustering_weights(self):
        """Load declustering weights from registry if available."""
        self._declust_weights = None
        self._using_declustered = False
        
        reg = self.get_registry()
        if not reg:
            return
        
        try:
            declust_results = reg.get_declustering_results()
            if declust_results:
                weighted_df = declust_results.get('weighted_dataframe')
                if weighted_df is not None and 'declust_weight' in weighted_df.columns:
                    self._declust_weights = weighted_df['declust_weight']
                    self._using_declustered = True
                    logger.info(f"LINEAGE: Loaded declustering weights ({len(self._declust_weights)} samples)")
        except Exception as e:
            logger.debug(f"Could not load declustering weights: {e}")
    
    def _on_declustering_loaded(self, results):
        """Handle declustering results loaded from registry."""
        self._load_declustering_weights()
        self._update_lineage_banner()
        # Re-run statistics with new weights
        if self._df is not None:
            self._run_stats()

    def _run_stats(self):
        """
        Compute and display statistics with lineage enforcement.
        
        LINEAGE GATES (STAT-005):
        - Warns if validation failed (but does not block)
        - Warns if using raw assays
        
        WEIGHTING (STAT-002):
        - Applies declustering weights when available
        - Shows both raw and weighted statistics for comparison
        """
        if self._df is None:
            return
        col = self.col_combo.currentText()
        if not col or col not in self._df.columns:
            return
        
        # Get values, dropping NaN
        vals = self._df[col].dropna()
        if vals.empty:
            self.stats_text.setText("No data available")
            return
        
        # STAT-002: Get declustering weights aligned with values
        weights = None
        if self._using_declustered and self._declust_weights is not None:
            try:
                # Align weights with values index
                weights = self._declust_weights.reindex(vals.index).dropna()
                if len(weights) != len(vals):
                    # If alignment fails, find common indices
                    common_idx = vals.index.intersection(self._declust_weights.index)
                    if len(common_idx) > 0:
                        vals = vals.loc[common_idx]
                        weights = self._declust_weights.loc[common_idx]
                    else:
                        weights = None
                        logger.warning("Could not align declustering weights with data")
            except Exception as e:
                weights = None
                logger.warning(f"Failed to apply declustering weights: {e}")
        
        # 1. Basic Stats
        txt = f"=== Statistics for {col} ===\n"
        txt += f"Data Source: {self._data_source_type.upper()}\n"
        txt += f"Validation: {self._validation_status}\n"
        txt += f"Declustered: {'Yes' if weights is not None else 'No'}\n"
        txt += "-" * 35 + "\n"
        
        # LINEAGE WARNING for validation issues
        if self._validation_status == "FAIL":
            txt += "⚠️ WARNING: Validation issues detected.\n"
            txt += "Review QC panel before using results.\n"
            txt += "-" * 35 + "\n"
        
        # Raw (unweighted) statistics
        count = len(vals)
        raw_mean = vals.mean()
        raw_std = vals.std()
        raw_var = vals.var()
        raw_cv = (raw_std / raw_mean * 100) if raw_mean != 0 else 0
        
        txt += f"Count:      {count:,}\n"
        txt += f"Mean (raw): {raw_mean:.4f}\n"
        txt += f"Std (raw):  {raw_std:.4f}\n"
        txt += f"CV (raw):   {raw_cv:.1f}%\n"
        txt += f"Min:        {vals.min():.4f}\n"
        txt += f"Max:        {vals.max():.4f}\n"
        txt += f"P25:        {vals.quantile(0.25):.4f}\n"
        txt += f"P50:        {vals.quantile(0.50):.4f}\n"
        txt += f"P75:        {vals.quantile(0.75):.4f}\n"
        
        # STAT-002: Weighted statistics if declustering available
        if weights is not None and len(weights) > 0:
            txt += "-" * 35 + "\n"
            txt += "DECLUSTERED STATISTICS:\n"
            
            # Weighted mean
            weight_sum = weights.sum()
            if weight_sum > 0:
                weighted_mean = (vals * weights).sum() / weight_sum
                weighted_var = (weights * (vals - weighted_mean) ** 2).sum() / weight_sum
                weighted_std = np.sqrt(weighted_var)
                weighted_cv = (weighted_std / weighted_mean * 100) if weighted_mean != 0 else 0
                
                txt += f"Mean (wtd): {weighted_mean:.4f}\n"
                txt += f"Std (wtd):  {weighted_std:.4f}\n"
                txt += f"CV (wtd):   {weighted_cv:.1f}%\n"
                
                # Bias correction
                bias = weighted_mean - raw_mean
                bias_pct = (bias / raw_mean * 100) if raw_mean != 0 else 0
                txt += f"Bias:       {bias:+.4f} ({bias_pct:+.1f}%)\n"
        
        # LINEAGE WARNING
        if self._data_source_type == "assays":
            txt += "-" * 35 + "\n"
            txt += "⚠️ WARNING: Using raw assays.\n"
            txt += "Statistics may be biased.\n"
            txt += "Consider compositing first.\n"
        
        self.stats_text.setText(txt)
        
        # 2. Grade Tonnage (Vectorized)
        self._compute_grade_tonnage(vals, weights)
    
    def _compute_grade_tonnage(self, vals: pd.Series, weights: Optional[pd.Series]):
        """
        Compute grade-tonnage curve with optional declustering weights.
        
        STAT-008: Tonnage Assumptions:
        - Tonnage = interval length (metres) if no density
        - For true tonnage: tonnage = length × density
        - Cross-sectional area assumed to be 1 m² (standard for drillholes)
        """
        try:
            cutoffs = [float(x.strip()) for x in self.cutoffs_edit.text().split(',') if x.strip()]
            cutoffs.sort()
            
            self.gt_table.setRowCount(0)
            
            # Pre-calculate interval lengths (STAT-008)
            # Try various column name conventions
            lengths = None
            for depth_to_col, depth_from_col in [
                ('depth_to', 'depth_from'),
                ('DEPTH_TO', 'DEPTH_FROM'),
                ('TO', 'FROM'),
                ('to', 'from')
            ]:
                if depth_to_col in self._df.columns and depth_from_col in self._df.columns:
                    lengths = self._df[depth_to_col] - self._df[depth_from_col]
                    break
            
            if lengths is None:
                # Fallback: assume unit length
                lengths = pd.Series(1.0, index=self._df.index)
                logger.debug("No depth columns found, using unit length for tonnage")
            
            # Align lengths with values index
            lengths = lengths.reindex(vals.index).fillna(1.0)
            
            # Check for density column (STAT-008)
            density = None
            for dens_col in ['density', 'DENSITY', 'Density', 'SG', 'sg']:
                if dens_col in self._df.columns:
                    density = self._df[dens_col].reindex(vals.index).fillna(2.7)  # Default 2.7 t/m³
                    break
            
            if density is not None:
                # True tonnage calculation: length × density × area (1 m²)
                tonnage_factor = lengths * density
                logger.debug("Using density-adjusted tonnage")
            else:
                # Length proxy (documented assumption)
                tonnage_factor = lengths
            
            for cut in cutoffs:
                mask = vals >= cut
                if not mask.any():
                    continue
                
                mask_idx = vals[mask].index
                
                # STAT-006: Guard against division by zero
                tonnes_above = tonnage_factor.loc[mask_idx].sum()
                if tonnes_above <= 0:
                    logger.debug(f"Skipping cutoff {cut}: zero tonnage above cutoff")
                    continue
                
                # Calculate grade with proper weighting
                # STAT-002: Apply declustering weights if available
                weighted_indicator = "✓" if weights is not None else ""
                
                if weights is not None:
                    # Combined weighting: length × declust_weight
                    combined_weights = tonnage_factor.loc[mask_idx] * weights.reindex(mask_idx).fillna(1.0)
                    weight_sum = combined_weights.sum()
                    if weight_sum > 0:
                        grade = (vals[mask] * combined_weights).sum() / weight_sum
                    else:
                        grade = (vals[mask] * tonnage_factor.loc[mask_idx]).sum() / tonnes_above
                else:
                    # Length-weighted grade
                    grade = (vals[mask] * tonnage_factor.loc[mask_idx]).sum() / tonnes_above
                
                metal = tonnes_above * grade
                
                row = self.gt_table.rowCount()
                self.gt_table.insertRow(row)
                self.gt_table.setItem(row, 0, QTableWidgetItem(str(cut)))
                self.gt_table.setItem(row, 1, QTableWidgetItem(f"{tonnes_above:.1f}"))
                self.gt_table.setItem(row, 2, QTableWidgetItem(f"{grade:.4f}"))
                self.gt_table.setItem(row, 3, QTableWidgetItem(f"{metal:.1f}"))
                self.gt_table.setItem(row, 4, QTableWidgetItem(weighted_indicator))
                
        except ValueError as e:
            logger.warning(f"Invalid cutoff values: {e}")

    def _export(self):
        """
        Export statistics report with full provenance metadata.
        
        Includes lineage information for JORC/SAMREC audit trail.
        """
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", "", "CSV (*.csv)")
        if path:
            from datetime import datetime
            
            # Export GT table content with provenance
            rows = []
            for r in range(self.gt_table.rowCount()):
                row_data = {
                    "Cutoff": self.gt_table.item(r, 0).text() if self.gt_table.item(r, 0) else "",
                    "Tonnage": self.gt_table.item(r, 1).text() if self.gt_table.item(r, 1) else "",
                    "Grade": self.gt_table.item(r, 2).text() if self.gt_table.item(r, 2) else "",
                    "Metal": self.gt_table.item(r, 3).text() if self.gt_table.item(r, 3) else "",
                    "Declustered": self.gt_table.item(r, 4).text() if self.gt_table.item(r, 4) else "",
                }
                rows.append(row_data)
            
            df_export = pd.DataFrame(rows)
            
            # Add provenance metadata as comment header
            provenance_header = [
                f"# GeoX Drillhole Statistics Report",
                f"# Generated: {datetime.now().isoformat()}",
                f"# Data Source: {self._data_source_type}",
                f"# Validation Status: {self._validation_status}",
                f"# Declustering Applied: {self._using_declustered}",
                f"# Sample Count: {len(self._df) if self._df is not None else 0}",
                f"# Element: {self.col_combo.currentText()}",
                f"#",
                f"# ASSUMPTIONS:",
                f"# - Tonnage = interval_length (metres) if no density column",
                f"# - If density present: tonnage = length × density",
                f"# - Cross-sectional area assumed 1 m²",
                f"#",
            ]
            
            # Write with provenance header
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(provenance_header) + '\n')
            
            # Append data
            df_export.to_csv(path, mode='a', index=False)
            
            # Also export summary statistics to separate file
            stats_path = path.replace('.csv', '_summary.csv')
            stats_rows = []
            col = self.col_combo.currentText()
            if col and self._df is not None and col in self._df.columns:
                vals = self._df[col].dropna()
                if not vals.empty:
                    stats_rows.append({"Metric": "Data_Source", "Value": self._data_source_type})
                    stats_rows.append({"Metric": "Validation_Status", "Value": self._validation_status})
                    stats_rows.append({"Metric": "Declustered", "Value": str(self._using_declustered)})
                    stats_rows.append({"Metric": "Count", "Value": len(vals)})
                    stats_rows.append({"Metric": "Mean_Raw", "Value": f"{vals.mean():.6f}"})
                    stats_rows.append({"Metric": "Std_Raw", "Value": f"{vals.std():.6f}"})
                    stats_rows.append({"Metric": "Min", "Value": f"{vals.min():.6f}"})
                    stats_rows.append({"Metric": "Max", "Value": f"{vals.max():.6f}"})
                    stats_rows.append({"Metric": "P25", "Value": f"{vals.quantile(0.25):.6f}"})
                    stats_rows.append({"Metric": "P50", "Value": f"{vals.quantile(0.50):.6f}"})
                    stats_rows.append({"Metric": "P75", "Value": f"{vals.quantile(0.75):.6f}"})
                    
                    # Add weighted stats if available
                    if self._using_declustered and self._declust_weights is not None:
                        try:
                            common_idx = vals.index.intersection(self._declust_weights.index)
                            if len(common_idx) > 0:
                                w = self._declust_weights.loc[common_idx]
                                v = vals.loc[common_idx]
                                w_sum = w.sum()
                                if w_sum > 0:
                                    w_mean = (v * w).sum() / w_sum
                                    w_var = (w * (v - w_mean) ** 2).sum() / w_sum
                                    w_std = np.sqrt(w_var)
                                    stats_rows.append({"Metric": "Mean_Declustered", "Value": f"{w_mean:.6f}"})
                                    stats_rows.append({"Metric": "Std_Declustered", "Value": f"{w_std:.6f}"})
                                    stats_rows.append({"Metric": "Bias_Correction", "Value": f"{w_mean - vals.mean():.6f}"})
                        except Exception as e:
                            logger.debug(f"Could not compute weighted stats for export: {e}")
            
            if stats_rows:
                pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
            
            QMessageBox.information(
                self, "Export Complete", 
                f"Saved to:\n{path}\n\nSummary statistics saved to:\n{stats_path}"
            )
            self._update_status(f"✅ Exported to {path}", "green")
