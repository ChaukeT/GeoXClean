"""
BLOCK MODEL IMPORT PANEL

Purpose: Load CSV block model data with column mapping and auto-detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QSplitter, QTextEdit, QVBoxLayout,
    QWidget, QSizePolicy, QFrame
)

from .base_analysis_panel import BaseAnalysisPanel
from .modern_widgets import (
    FileInputCard, ModernProgressBar, SectionHeader, 
    StatusBadge, ActionButton, Colors
)

logger = logging.getLogger(__name__)


class BlockModelImportPanel(BaseAnalysisPanel):
    """Panel for loading block model CSV files with column mapping."""

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="block_model_import")
        self.setWindowTitle("Block Model Loader")
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
        self.file_path = None
        self.df = None
        self.column_mapping = None
        self.selected_properties = None

    def _get_main_window(self):
        """Get reference to main window to close its progress dialog if needed."""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'progress_dialog'):
                return parent
            parent = parent.parent() if hasattr(parent, 'parent') else None
        return None

    def _build_ui(self):
        # Apply modern base styling
        self.setStyleSheet(self._get_stylesheet())

        # Use the main_layout provided by BaseAnalysisPanel
        main_layout = self.main_layout
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        header = SectionHeader(
            "Import Block Model",
            "Load CSV block model with automatic column detection"
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
        files_label = QLabel("SOURCE FILE")
        files_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        l_layout.addWidget(files_label)
        
        # File input card
        self.file_card = FileInputCard(
            label="Block Model CSV",
            file_filter="CSV Files (*.csv);;All Files (*)",
            required=True,
            icon="🧱"
        )
        self.file_card.fileSelected.connect(self._on_file_selected)
        self.file_card.fileCleared.connect(self._on_file_cleared)
        l_layout.addWidget(self.file_card)
        
        l_layout.addStretch()
        
        # Model info summary
        self.info_frame = QFrame()
        self.info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_PRIMARY};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
            }}
        """)
        self.info_frame.hide()
        
        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(12, 10, 12, 10)
        info_layout.setSpacing(8)
        
        info_title = QLabel("Model Summary")
        info_title.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 10px; font-weight: 600;")
        info_layout.addWidget(info_title)
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet(f"color: {Colors.TEXT_PRIMARY}; font-size: 11px;")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        l_layout.addWidget(self.info_frame)
        
        # Load button
        self.btn_load = ActionButton("Load Block Model", variant="primary", icon="🚀")
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self._start_load)
        l_layout.addWidget(self.btn_load)
        
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
        
        # Log area with dark terminal theme
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

    def _on_file_selected(self, path: str):
        """Handle file selection from card - open column mapping dialog."""
        try:
            logger.info(f"Loading block model file: {path}")
            self.file_path = Path(path)
            
            # Read CSV file
            df_raw = pd.read_csv(path)
            logger.info(f"Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
            
            # Show column mapping dialog
            from .block_model_column_mapping_dialog import BlockModelColumnMappingDialog
            
            dialog = BlockModelColumnMappingDialog(df_raw, self)
            
            if dialog.exec() == dialog.DialogCode.Accepted:
                # User accepted the mapping
                self.column_mapping = dialog.get_mapping()
                self.selected_properties = dialog.get_selected_properties()
                
                logger.info(f"Column mapping accepted: {self.column_mapping}")
                logger.info(f"Selected properties: {sorted(self.selected_properties)}")
                
                # Update card with row count
                self.file_card.set_row_count(len(df_raw))
                
                # Update info panel
                self.info_label.setText(
                    f"Rows: {len(df_raw):,}\n"
                    f"Columns: {len(df_raw.columns)}\n"
                    f"Properties: {len(self.selected_properties)}"
                )
                self.info_frame.show()
                
                # Log the import
                self._log(f"✓ File loaded: {Path(path).name}", "success")
                self._log(f"  {len(df_raw):,} rows, {len(df_raw.columns)} columns", "info")
                
                if self.column_mapping:
                    mapping_str = ", ".join([f"{k}→{v}" for k, v in self.column_mapping.items()])
                    self._log(f"  Mapping: {mapping_str}", "info")
                
                if self.selected_properties:
                    self._log(f"  Properties: {len(self.selected_properties)} selected", "info")
                
                # Enable load button
                self.btn_load.setEnabled(True)
            else:
                # User cancelled the mapping dialog
                logger.info("Column mapping cancelled")
                self.file_card._on_clear()
                self._log("Column mapping cancelled", "warning")
                self.file_path = None
                
        except Exception as e:
            logger.error(f"Error loading file: {e}", exc_info=True)
            self.file_card._on_clear()
            self._log(f"✗ Failed to load file: {e}", "error")
            QMessageBox.critical(self, "Import Error", f"Failed to load file:\n{e}")

    def _on_file_cleared(self):
        """Handle file clear from card."""
        self.file_path = None
        self.column_mapping = None
        self.selected_properties = None
        self.info_frame.hide()
        self.btn_load.setEnabled(False)
        self._log("File cleared", "info")

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

    def _start_load(self):
        """Start block model loading directly without triggering main window's progress dialog."""
        try:
            logger.info("=" * 80)
            logger.info("BLOCK MODEL IMPORT: Starting load from UI")
            logger.info("=" * 80)
            
            if not self.file_path:
                QMessageBox.warning(self, "Error", "No file selected.")
                return
            
            self.btn_load.setEnabled(False)
            self.progress_bar.show()
            self.progress_bar.setValue(0)
            self.progress_bar.setLabel("Loading block model...")
            self.status_badge.setText("Loading")
            self.status_badge.set_state(StatusBadge.State.INFO)
            
            self._log("─" * 40, "info")
            self._log("Starting block model load...", "info")
            
            # Convert file_path to Path object
            file_path_obj = Path(self.file_path) if not isinstance(self.file_path, Path) else self.file_path
            
            logger.info(f"Loading file: {file_path_obj}")
            
            # Parse file directly using parser registry
            self._log("> Parsing file...", "info")
            
            try:
                from ..parsers import parser_registry
                
                # Build kwargs for parser with column mappings
                parse_kwargs = {
                    'x_col': self.column_mapping.get('X'),
                    'y_col': self.column_mapping.get('Y'),
                    'z_col': self.column_mapping.get('Z'),
                    'dx_col': self.column_mapping.get('DX'),
                    'dy_col': self.column_mapping.get('DY'),
                    'dz_col': self.column_mapping.get('DZ'),
                }
                # Remove None values
                parse_kwargs = {k: v for k, v in parse_kwargs.items() if v is not None}
                
                logger.info(f"Parse kwargs: {parse_kwargs}")
                
                self.progress_bar.setValue(30)
                self.progress_bar.setLabel("Parsing block geometry...")
                
                block_model = parser_registry.parse_file(file_path_obj, **parse_kwargs)
                
                self._log(f"> Loaded {block_model.block_count:,} blocks", "success")
                
                self.progress_bar.setValue(60)
                self.progress_bar.setLabel("Validating model...")
                
                # Validate
                self._log("> Validating model...", "info")
                errors = block_model.validate()
                if errors:
                    raise ValueError(f"Validation errors: {', '.join(errors)}")
                
                self.progress_bar.setValue(90)
                self.progress_bar.setLabel("Registering model...")
                
                # Call completion handler with result
                result = {
                    "block_model": block_model,
                    "file_path": str(file_path_obj)
                }
                self._on_load_complete(result)
                
            except Exception as e:
                logger.error(f"Parse error: {e}", exc_info=True)
                self._on_load_complete({"error": str(e)})
                
        except Exception as e:
            logger.error(f"ERROR in _start_load: {e}", exc_info=True)
            self._log(f"✗ Failed to start load: {e}", "error")
            self.status_badge.setText("Failed")
            self.status_badge.set_state(StatusBadge.State.ERROR)
            QMessageBox.critical(self, "Load Error", f"Failed to start load:\n{e}")
            self.btn_load.setEnabled(True)
            self.progress_bar.hide()

    def _on_load_complete(self, result: Dict[str, Any]):
        """Handle completion of block model load task."""
        try:
            self.progress_bar.hide()
            self.btn_load.setEnabled(True)
            
            if result is None:
                self._log("✗ Load returned no result", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Load Error", "Load returned no result.")
                return
            
            if result.get("error"):
                error_msg = result["error"]
                self._log(f"✗ Load error: {error_msg}", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Load Error", error_msg)
                return
            
            # Extract block model from result
            block_model = result.get("block_model")
            
            if block_model is None:
                self._log("✗ No block model in result", "error")
                self.status_badge.setText("Failed")
                self.status_badge.set_state(StatusBadge.State.ERROR)
                QMessageBox.critical(self, "Load Error", "No block model in result.")
                return
            
            prop_count = len(block_model.get_property_names())
            
            self._log("─" * 40, "info")
            self._log(f"✓ Block model loaded successfully!", "success")
            self._log(f"  Blocks: {block_model.block_count:,}", "success")
            self._log(f"  Properties: {prop_count}", "success")
            
            self.status_badge.setText("Complete")
            self.status_badge.set_state(StatusBadge.State.SUCCESS)
            
            # Register block model with DataRegistry
            try:
                reg = self.get_registry()
                if reg is None:
                    raise ValueError("DataRegistry not available")
                
                # Close any existing progress dialog from main window
                main_window = self._get_main_window()
                if main_window and hasattr(main_window, 'progress_dialog') and main_window.progress_dialog:
                    try:
                        main_window.progress_dialog.close()
                        main_window.progress_dialog = None
                    except Exception as e:
                        logger.error(f"Failed to close progress dialog: {e}", exc_info=True)
                
                reg.register_block_model(
                    block_model,
                    source_panel="Block Model Loader",
                    metadata={
                        "file_path": str(self.file_path),
                        "column_mapping": self.column_mapping
                    }
                )
                
                self._log("✓ Block model registered and displayed", "success")
                
                # Defer success message
                def show_success():
                    QMessageBox.information(
                        self, 
                        "Success", 
                        f"Block model loaded successfully!\n\n"
                        f"Blocks: {block_model.block_count:,}\n"
                        f"Properties: {prop_count}"
                    )
                QTimer.singleShot(300, show_success)
                
            except Exception as e:
                logger.error(f"Failed to register block model: {e}", exc_info=True)
                self._log(f"⚠️ Could not register with DataRegistry: {e}", "warning")
                QMessageBox.warning(
                    self, 
                    "Partial Success", 
                    f"Block model loaded but could not be registered:\n{e}"
                )
                
        except Exception as e:
            logger.error(f"ERROR in _on_load_complete: {e}", exc_info=True)
            self._log(f"✗ Unexpected error: {e}", "error")
            self.status_badge.setText("Failed")
            self.status_badge.set_state(StatusBadge.State.ERROR)
            self.btn_load.setEnabled(True)
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