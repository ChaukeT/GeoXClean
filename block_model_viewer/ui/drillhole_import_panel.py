"""
DRILLHOLE IMPORT PANEL

Purpose: Load raw CSV data (Collars, Surveys, Assays, Lithology, Structures), 
compute 3D coordinates (Desurveying), and register data to the application.

Structural data can be imported with dip/dip-direction or alpha/beta orientations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, QTimer
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QProgressBar, QSplitter, QTextEdit, QVBoxLayout,
    QWidget, QSizePolicy, QDialog, QFrame, QScrollArea
)

from .base_analysis_panel import BaseAnalysisPanel
from .modern_widgets import (
    FileInputCard, ModernProgressBar, SectionHeader, 
    StatusBadge, ActionButton, Colors
)

logger = logging.getLogger(__name__)


# Worker logic moved to DataController._prepare_drillhole_import_payload
# This ensures pure computation with no access to DataRegistry or Qt objects


# --- MAIN PANEL ---
class DrillholeImportPanel(BaseAnalysisPanel):
    task_name = "drillhole_import"

    # PanelManager metadata
    PANEL_ID = "DrillholeImportPanel"
    PANEL_NAME = "DrillholeImport Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="drillhole_import")
        self.setWindowTitle("Drillhole Data Loader")
        # Set minimum size when embedded in a dialog
        if parent is not None:
            self.setMinimumSize(900, 650)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._init_state()
        self._build_ui()



    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
            QWidget {{
                font-family: 'Segoe UI', -apple-system, sans-serif;
                color: {Colors.TEXT_PRIMARY};
                background-color: {Colors.BG_PRIMARY};
            }}
            QGroupBox {{
                background-color: {Colors.BG_SURFACE};
                border: 1px solid {Colors.BORDER};
                border-radius: 12px;
                margin-top: 16px;
                padding-top: 12px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                color: {Colors.TEXT_PRIMARY};
            }}
        """

    def refresh_theme(self):
        """Update colors when theme changes."""
        # Rebuild stylesheet with new theme colors
        self.setStyleSheet(self._get_stylesheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()

    def _init_state(self):
        self.collar_df = None
        self.survey_df = None
        self.assay_df = None
        self.lithology_df = None
        self.structures_df = None
        self.upload_thread = None
        self.file_cards = {}

    def _build_ui(self):
        # Apply modern base styling
        self.setStyleSheet(self._get_stylesheet())
        
        # Clear existing layout from BaseAnalysisPanel
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
        
        # Create new main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = SectionHeader(
            "Import Drillhole Data",
            "Load collar, survey, assay, lithology, and structural CSV files"
        )
        main_layout.addWidget(header)
        
        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
        """)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # === LEFT PANEL: File Selection ===
        left_widget = QFrame()
        left_widget.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")
        left_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_widget.setMinimumWidth(380)
        left_widget.setMaximumWidth(450)
        
        l_layout = QVBoxLayout(left_widget)
        l_layout.setContentsMargins(20, 20, 20, 20)
        l_layout.setSpacing(16)
        
        # Section label
        files_label = QLabel("SOURCE FILES")
        files_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        l_layout.addWidget(files_label)
        
        # File input cards
        file_types = [
            ("collar", "Collar Data", "📍", True),
            ("survey", "Survey Data", "🧭", False),
            ("assay", "Assay Data", "🧪", False),
            ("lithology", "Lithology Data", "🪨", False),
            ("structures", "Structural Data", "📐", False),
        ]
        
        for ftype, label, icon, required in file_types:
            card = FileInputCard(
                label=label,
                file_filter="CSV Files (*.csv);;All Files (*)",
                required=required,
                icon=icon
            )
            card.fileSelected.connect(lambda p, t=ftype: self._on_file_selected(t, p))
            card.fileCleared.connect(lambda t=ftype: self._on_file_cleared(t))
            self.file_cards[ftype] = card
            l_layout.addWidget(card)
        
        l_layout.addStretch()
        
        # Import stats summary
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)
        self.stats_frame.hide()
        
        stats_layout = QVBoxLayout(self.stats_frame)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(4)
        
        stats_title = QLabel("Import Summary")
        stats_title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px; font-weight: 600;")
        stats_layout.addWidget(stats_title)
        
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 11px;")
        stats_layout.addWidget(self.stats_label)
        
        l_layout.addWidget(self.stats_frame)
        
        # Build button
        self.btn_upload = ActionButton("Build Database", variant="primary", icon="🔨")
        self.btn_upload.setEnabled(False)
        self.btn_upload.clicked.connect(self._start_upload)
        l_layout.addWidget(self.btn_upload)
        
        splitter.addWidget(left_widget)
        
        # === RIGHT PANEL: Log & Progress ===
        right_widget = QFrame()
        right_widget.setStyleSheet(f"background-color: {Colors.BG_SURFACE};")
        right_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        r_layout = QVBoxLayout(right_widget)
        r_layout.setContentsMargins(20, 20, 20, 20)
        r_layout.setSpacing(12)
        
        # Log header
        log_header = QHBoxLayout()
        log_title = QLabel("Process Log")
        log_title.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 13px;
            font-weight: 600;
        """)
        log_header.addWidget(log_title)
        
        self.status_badge = StatusBadge("Ready", StatusBadge.State.NEUTRAL)
        log_header.addStretch()
        log_header.addWidget(self.status_badge)
        
        r_layout.addLayout(log_header)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet(f"""
            QTextEdit {{
                background-color: {ModernColors.PANEL_BG};
                color: #e2e8f0;
                border: 1px solid {{Colors.BORDER}};
                border-radius: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                padding: 12px;
            }}
        """)
        self.log_area.setPlaceholderText("Import log will appear here...")
        r_layout.addWidget(self.log_area)
        
        # Progress bar
        self.progress_bar = ModernProgressBar()
        self.progress_bar.hide()
        r_layout.addWidget(self.progress_bar)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)

    def _on_file_selected(self, file_type: str, path: str):
        """Handle file selection from card."""
        try:
            logger.info(f"Loading {file_type} file: {path}")
            df_raw = pd.read_csv(path)

            # BUG FIX #17: Validate DataFrame was loaded successfully
            if df_raw is None:
                raise ValueError("Failed to parse CSV file - returned None")
            if df_raw.empty:
                raise ValueError("CSV file is empty or contains no valid data")

            logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")

            # Show column mapping dialog
            from .column_mapping_dialog import ColumnMappingDialog

            dialog = ColumnMappingDialog(df_raw, file_type, self)
            
            if dialog.exec() == dialog.DialogCode.Accepted:
                df = dialog.get_mapped_dataframe()
                mapping = dialog.get_mapping()
                
                logger.info(f"Column mapping accepted: {mapping}")
                
                # Store the mapped DataFrame
                setattr(self, f"{file_type}_df", df)
                
                # Update card with row count
                self.file_cards[file_type].set_row_count(len(df))
                
                # Log the import
                self._log(f"✓ Loaded {file_type}: {len(df):,} rows from {Path(path).name}", "success")
                
                if mapping:
                    mapping_str = ", ".join([f"{k}→{v}" for k, v in mapping.items()])
                    self._log(f"  Mapping: {mapping_str}", "info")
                
                # Update stats and button state
                self._update_stats()
                self._check_upload_enabled()
                
                # Auto-detect related files when collar is loaded
                if file_type == "collar":
                    self._auto_detect_related_files(Path(path))
            else:
                logger.info(f"Column mapping cancelled for {file_type}")
                self.file_cards[file_type]._on_clear()
                self._log(f"Cancelled {file_type} import", "warning")
                
        except Exception as e:
            logger.error(f"Error loading {file_type} file: {e}", exc_info=True)
            self.file_cards[file_type]._on_clear()
            self._log(f"✗ Failed to load {file_type}: {e}", "error")
            QMessageBox.critical(self, "Import Error", f"Failed to load {file_type} file:\n{e}")

    def _auto_detect_related_files(self, collar_path: Path):
        """
        Auto-detect survey, assay, and lithology files from the same folder.
        
        Uses multiple detection strategies like Leapfrog:
        1. Filename patterns: files containing 'survey', 'assay', 'lithology', etc.
        2. Column inspection: reads CSV headers to identify file types by column names
        """
        folder = collar_path.parent
        collar_stem = collar_path.stem.lower()
        
        # Define search patterns for each file type
        # Filename patterns
        filename_patterns = {
            "survey": ["survey", "surveys", "deviation", "dev"],
            "assay": ["assay", "assays", "grades", "grade", "samples", "sample"],
            "lithology": ["lithology", "litho", "lith", "geology", "geo", "rock", "rocktype"],
            "structures": ["structure", "structures", "structural", "televiewer", "optv", "actv", "oriented", "discontinuity", "joints", "faults"],
        }
        
        # Column name patterns - columns that indicate file type
        # These are checked if filename matching doesn't find a file
        # Note: _column_matches() handles unit suffixes like _m, _deg, _ft automatically
        column_patterns = {
            "survey": {
                # Survey files have directional data (azimuth/dip) at depth points
                "required": ["azimuth", "azi", "azim", "bearing", "brg", "dip", "incl", "inclination"],
                "optional": ["depth", "at", "survey"],
                "exclude": [],  # Survey can have depth/from column
            },
            "assay": {
                # Assay files have from/to intervals with grade values
                "required": ["from", "to", "start", "end"],
                "indicators": ["au", "ag", "cu", "fe", "pb", "zn", "ni", "co", "mn", "grade", "assay", "sample", "ppm", "ppb", "pct", "percent"],
                "exclude": ["azimuth", "azi", "dip", "incl", "lithology", "lith", "rock", "geology"],
            },
            "lithology": {
                # Lithology files have from/to intervals with rock type descriptions
                "required": ["from", "to", "start", "end"],
                "indicators": ["lith", "lithology", "rock", "rocktype", "geology", "geo", "code", "description", "desc", "unit", "formation", "strat"],
                "exclude": ["azimuth", "azi", "dip", "incl", "au", "ag", "cu", "grade", "ppm"],
            },
            "structures": {
                # Structural data has orientation measurements (dip/dip-direction or alpha/beta)
                "required": ["from", "to", "start", "end"],
                "indicators": ["dip_direction", "dipdir", "dip_dir", "dd", "alpha", "beta", "structure", "feature_type", "joint", "fault", "bedding", "foliation"],
                "exclude": ["au", "ag", "cu", "grade", "lithology", "lith", "rock"],
            },
        }
        
        detected_files = {}
        
        # Get all CSV files in the folder
        try:
            csv_files = list(folder.glob("*.csv")) + list(folder.glob("*.CSV"))
        except Exception as e:
            logger.warning(f"Could not scan folder for related files: {e}")
            return
        
        # Extract prefix from collar file (remove collar-related keywords)
        collar_keywords = ["collar", "collars", "col"]
        prefix = collar_stem
        for kw in collar_keywords:
            prefix = prefix.replace(kw, "").strip("_- ")
        
        # First pass: match by filename
        for file_type, patterns in filename_patterns.items():
            # Skip if already loaded
            if getattr(self, f"{file_type}_df", None) is not None:
                continue
            
            best_match = None
            best_score = 0
            
            for csv_file in csv_files:
                if csv_file == collar_path:
                    continue
                    
                file_stem = csv_file.stem.lower()
                score = 0
                
                # Check if file contains any of the patterns
                for pattern in patterns:
                    if pattern in file_stem:
                        score += 10
                        break
                
                if score == 0:
                    continue
                
                # Bonus points for matching prefix
                if prefix and prefix in file_stem:
                    score += 5
                
                # Bonus for similar length (likely same naming convention)
                len_diff = abs(len(file_stem) - len(collar_stem))
                if len_diff < 5:
                    score += 2
                
                if score > best_score:
                    best_score = score
                    best_match = csv_file
            
            if best_match:
                detected_files[file_type] = best_match
        
        # Second pass: inspect columns for files not detected by filename
        undetected_types = [ft for ft in filename_patterns.keys() 
                           if ft not in detected_files and getattr(self, f"{ft}_df", None) is None]
        
        if undetected_types:
            logger.info(f"Checking column headers for undetected types: {undetected_types}")
            
            # Files not yet assigned
            unassigned_files = [f for f in csv_files 
                               if f != collar_path and f not in detected_files.values()]
            
            for csv_file in unassigned_files:
                try:
                    # Read only the header row to inspect columns
                    df_header = pd.read_csv(csv_file, nrows=0)
                    columns_original = list(df_header.columns)
                    
                    logger.info(f"Inspecting {csv_file.name}: columns = {columns_original}")
                    
                    file_type = self._identify_file_type_by_columns(columns_original, column_patterns, undetected_types)
                    
                    if file_type and file_type not in detected_files:
                        detected_files[file_type] = csv_file
                        undetected_types.remove(file_type)
                        logger.info(f"✓ Detected {file_type} by column inspection: {csv_file.name}")
                    else:
                        logger.debug(f"  No match for {csv_file.name}")
                        
                except Exception as e:
                    logger.warning(f"Could not inspect columns of {csv_file.name}: {e}")
                    continue
        
        # Report and auto-populate detected files
        if detected_files:
            self._log("─" * 40, "info")
            self._log("📁 Auto-detected related files:", "info")
            
            for file_type, file_path in detected_files.items():
                self._log(f"  • {file_type.title()}: {file_path.name}", "info")
                # Set the file path in the card without emitting signal
                # User needs to click "Browse" to confirm and map columns
                self.file_cards[file_type].set_file(str(file_path), emit_signal=False)
            
            self._log("  Click 'Browse' on each card to confirm", "info")

    def _normalize_column_name(self, col: str) -> str:
        """
        Normalize column name by removing units and standardizing separators.
        
        Examples:
            'azimuth_deg' -> 'azimuth'
            'from_m' -> 'from'
            'total depth m' -> 'totaldepth'
            'HOLE_ID' -> 'holeid'
        """
        import re
        col = col.lower().strip()
        
        # Remove common unit suffixes (with separator)
        unit_patterns = [
            r'[_\s]+(m|meters?|ft|feet|deg|degrees?|pct|percent|ppm|ppb)$',
            r'[_\s]+(metres?)$',
        ]
        for pattern in unit_patterns:
            col = re.sub(pattern, '', col)
        
        # Remove all separators for matching (but keep original for display)
        col_normalized = re.sub(r'[_\s\-]+', '', col)
        
        return col_normalized

    def _column_matches(self, column: str, keywords: list) -> bool:
        """
        Check if a column name matches any of the keywords.
        
        Handles variations like:
            - 'azimuth_deg' matches 'azimuth'
            - 'from_m' matches 'from'
            - 'total_depth' matches 'depth', 'totaldepth'
            - 'hole_id' matches 'holeid', 'hole'
        """
        col_normalized = self._normalize_column_name(column)
        col_lower = column.lower()
        
        for kw in keywords:
            kw_lower = kw.lower()
            # Direct substring match in original
            if kw_lower in col_lower:
                return True
            # Match in normalized (no separators)
            if kw_lower in col_normalized:
                return True
            # Keyword at start or end of normalized
            if col_normalized.startswith(kw_lower) or col_normalized.endswith(kw_lower):
                return True
        return False

    def _identify_file_type_by_columns(self, columns: list, patterns: dict, candidates: list) -> Optional[str]:
        """
        Identify file type by inspecting column names.
        
        Args:
            columns: List of lowercase column names
            patterns: Dict of column patterns for each file type
            candidates: List of file types to check
            
        Returns:
            Detected file type or None
        """
        best_type = None
        best_score = 0
        
        for file_type in candidates:
            if file_type not in patterns:
                continue
                
            pattern = patterns[file_type]
            score = 0
            
            # Check for required columns (need at least one)
            has_required = False
            for req in pattern.get("required", []):
                if any(self._column_matches(col, [req]) for col in columns):
                    has_required = True
                    score += 5
                    break
            
            if not has_required:
                continue
            
            # Check for indicator columns (file type specific)
            for ind in pattern.get("indicators", []):
                if any(self._column_matches(col, [ind]) for col in columns):
                    score += 3
            
            # Check for exclusion columns (likely NOT this type)
            for excl in pattern.get("exclude", []):
                if any(self._column_matches(col, [excl]) for col in columns):
                    score -= 5
            
            # Survey special case: must have azimuth/dip type columns
            if file_type == "survey":
                directional_keywords = ["azimuth", "azi", "bearing", "brg", "dip", "incl", "inclination"]
                has_directional = any(
                    self._column_matches(col, directional_keywords)
                    for col in columns
                )
                if not has_directional:
                    continue
                score += 10
            
            # Lithology special case: must have lithology/rock type column
            if file_type == "lithology":
                lith_keywords = ["lith", "lithology", "rock", "geology", "geo", "code", "type", "unit"]
                has_lith = any(
                    self._column_matches(col, lith_keywords)
                    for col in columns
                )
                if has_lith:
                    score += 8
            
            if score > best_score:
                best_score = score
                best_type = file_type
        
        # Only return if we have a reasonable confidence
        return best_type if best_score >= 5 else None

    def _on_file_cleared(self, file_type: str):
        """Handle file clear from card."""
        setattr(self, f"{file_type}_df", None)
        self._update_stats()
        self._check_upload_enabled()
        self._log(f"Cleared {file_type} file", "info")

    def _log(self, message: str, level: str = "info"):
        """Add a message to the log with styling."""
        colors = {
            "info": f"{ModernColors.TEXT_SECONDARY}",
            "success": "#34d399",
            "warning": "#fbbf24",
            "error": "#f87171",
        }
        color = colors.get(level, colors["info"])
        self.log_area.append(f'<span style="color: {color}">{message}</span>')

    def _update_stats(self):
        """Update the import summary statistics."""
        parts = []
        total_rows = 0
        
        if self.collar_df is not None:
            parts.append(f"Collars: {len(self.collar_df):,}")
            total_rows += len(self.collar_df)
        if self.survey_df is not None:
            parts.append(f"Surveys: {len(self.survey_df):,}")
            total_rows += len(self.survey_df)
        if self.assay_df is not None:
            parts.append(f"Assays: {len(self.assay_df):,}")
            total_rows += len(self.assay_df)
        if self.lithology_df is not None:
            parts.append(f"Lithology: {len(self.lithology_df):,}")
            total_rows += len(self.lithology_df)
        if self.structures_df is not None:
            parts.append(f"Structures: {len(self.structures_df):,}")
            total_rows += len(self.structures_df)
        
        if parts:
            self.stats_label.setText(" • ".join(parts))
            self.stats_frame.show()
        else:
            self.stats_frame.hide()

    def _check_upload_enabled(self):
        """Enable upload button if collar data exists."""
        self.btn_upload.setEnabled(self.collar_df is not None)

    def _start_upload(self):
        """Start drillhole import using controller.run_task() pipeline."""
        try:
            logger.info("=" * 80)
            logger.info("DRILLHOLE IMPORT: Starting upload from UI")
            logger.info("=" * 80)
            
            if not self.controller:
                logger.error("Controller not available")
                QMessageBox.warning(self, "Error", "Controller not available.")
                return
            
            self.btn_upload.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.progress_bar.setLabel("Preparing data...")
            self.status_badge.setText("Processing")
            self.status_badge.set_state(StatusBadge.State.INFO)
            
            self._log("─" * 40, "info")
            self._log("Starting database build...", "info")
            
            # Prepare params - copy data to avoid mutation
            try:
                params = {
                    'assay_df': self.assay_df.copy() if self.assay_df is not None else None,
                    'collar_df': self.collar_df.copy() if self.collar_df is not None else None,
                    'survey_df': self.survey_df.copy() if self.survey_df is not None else None,
                    'lithology_df': self.lithology_df.copy() if self.lithology_df is not None else None,
                    'structures_df': self.structures_df.copy() if self.structures_df is not None else None
                }
            except Exception as e:
                logger.error(f"ERROR copying DataFrames: {e}", exc_info=True)
                self._log(f"✗ Failed to prepare data: {e}", "error")
                self.btn_upload.setEnabled(True)
                self.progress_bar.hide()
                return
            
            # Progress callback to update UI
            def progress_callback(percent: int, message: str):
                try:
                    def update_ui():
                        self.progress_bar.setValue(percent)
                        self.progress_bar.setLabel(message)
                        self._log(f"  {message}", "info")
                    QTimer.singleShot(0, update_ui)
                except Exception as e:
                    logger.error(f"ERROR in progress callback: {e}")
            
            # Run task via controller
            self.controller.run_task(
                'drillhole_import',
                params,
                callback=self._on_import_complete,
                progress_callback=progress_callback
            )
            
        except Exception as e:
            logger.error(f"ERROR in _start_upload: {e}", exc_info=True)
            self._log(f"✗ Failed to start import: {e}", "error")
            QMessageBox.critical(self, "Import Error", f"Failed to start import:\n{e}")
            self.btn_upload.setEnabled(True)
            self.progress_bar.hide()

    def _on_import_complete(self, result: Dict[str, Any]):
        """Handle completion of drillhole import task."""
        try:
            self.progress_bar.hide()
            self.btn_upload.setEnabled(True)
            
            if result is None:
                self._log("✗ Import returned no result", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Import Error", "Import returned no result.")
                return
            
            if result.get("error"):
                error_msg = result["error"]
                self._log(f"✗ Import error: {error_msg}", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Import Error", error_msg)
                return
            
            # Extract data from result
            drillhole_data = result.get("drillhole_data")
            metadata = result.get("metadata")
            
            if drillhole_data is None:
                self._log("✗ No drillhole data in result", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Import Error", "No drillhole data in result.")
                return
            
            # Register to DataRegistry
            try:
                reg = self.get_registry()
                if reg is None:
                    raise ValueError("DataRegistry not available")
                
                reg.register_drillhole_data(
                    drillhole_data,
                    source_panel="Drillhole Loader",
                    metadata=metadata
                )
                
                self._log("─" * 40, "info")
                self._log("✓ Database built successfully!", "success")
                self._log("✓ 3D coordinates computed", "success")
                self._log("✓ Data registered to application", "success")
                
                self.status_badge.setText("Complete")
                self.status_badge.set_state(StatusBadge.State.SUCCESS)
                
                # Defer success message
                def show_success():
                    QMessageBox.information(
                        self,
                        "Success",
                        "Drillhole database built and coordinates computed.\n\n"
                        "Use the Drillhole Control panel to visualize the data."
                    )
                QTimer.singleShot(300, show_success)
                
            except Exception as e:
                logger.error(f"Failed to register drillhole data: {e}", exc_info=True)
                self._log(f"✗ Failed to register data: {e}", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Registration Error", f"Failed to register data:\n{e}")
            
        except Exception as e:
            logger.error(f"ERROR in _on_import_complete: {e}", exc_info=True)
            self._log(f"✗ Unexpected error: {e}", "error")
            self.status_badge.setText("Failed")
            self.status_badge.set_state(StatusBadge.State.ERROR)
            self.btn_upload.setEnabled(True)
            self.progress_bar.hide()

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save.
        
        Note: File paths are not saved as they are not portable across systems.
        Column mappings would only be useful if the same files are loaded.
        """
        # Import panels typically don't save settings since file paths are transient
        return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load.
        
        Import panels don't typically restore settings since the actual
        data files are saved with the project separately.
        """
        pass