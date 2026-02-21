"""
Drillhole Data Loader Panel

Dedicated Data Ingestion Interface for loading raw drillhole data.
Focus: Show data previews to verify column mapping, not validation errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, 
    QFrame, QTableWidget, QTableWidgetItem, 
    QHeaderView, QTabWidget, QProgressBar,
    QSplitter, QSizePolicy
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush, QFont

from .base_analysis_panel import BaseAnalysisPanel
from .modern_widgets import (
    FileInputCard, SectionHeader, StatusBadge, 
    ActionButton, ModernProgressBar, Colors
)

logger = logging.getLogger(__name__)


class DrillholePanel(BaseAnalysisPanel):
    """
    Dedicated Drillhole Data Loader.
    
    Functionality:
    1. Select source CSV/Excel files.
    2. Import into application memory.
    3. Preview raw data to verify column mapping.
    """
    # PanelManager metadata
    PANEL_ID = "DrillholePanel"
    PANEL_NAME = "Drillhole Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT

    task_name = "drillhole_import"

    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="drillhole_loader")
        self.setWindowTitle("Drillhole Loader")
        
        self.database = None
        self.file_cards = {}
        
        # Modern base styling
        self.setStyleSheet(f"""
            QWidget {{
                font-family: 'Segoe UI', -apple-system, sans-serif;
                color: {Colors.TEXT_PRIMARY};
            }}
            QFrame {{
                background-color: {Colors.BG_PRIMARY};
            }}
            QTableWidget {{
                background-color: {Colors.BG_SURFACE};
                gridline-color: {Colors.BORDER};
                border: 1px solid {Colors.BORDER};
                border-radius: 8px;
                selection-background-color: {Colors.BG_SELECTED};
            }}
            QTableWidget::item {{
                padding: 6px 8px;
            }}
            QHeaderView::section {{
                background-color: {Colors.BG_PRIMARY};
                padding: 8px 12px;
                border: none;
                border-right: 1px solid {Colors.BORDER};
                border-bottom: 1px solid {Colors.BORDER};
                font-weight: 600;
                font-size: 11px;
                color: {Colors.TEXT_SECONDARY};
            }}
        """)
        
        self._build_ui()
    
    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header
        header = self._create_header()
        main_layout.addWidget(header)

        # Main Content (Split View)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {Colors.BORDER};
            }}
        """)
        
        self.left_panel = self._create_control_panel()
        self.right_panel = self._create_preview_panel()
        
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(1, 4)
        
        main_layout.addWidget(splitter)

        # Footer
        main_layout.addWidget(self._create_footer())

    def _create_header(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SURFACE};
                border-bottom: 1px solid {Colors.BORDER};
            }}
        """)
        frame.setFixedHeight(70)
        
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(24, 0, 24, 0)

        # Title section
        title_layout = QVBoxLayout()
        title_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        title_layout.setSpacing(2)
        
        title = QLabel("Data Import")
        title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {Colors.TEXT_PRIMARY};
        """)
        
        subtitle = QLabel("Load raw drillhole data into the project")
        subtitle.setStyleSheet(f"""
            font-size: 12px;
            color: {Colors.TEXT_SECONDARY};
        """)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        # Status Badge
        self.lbl_status = StatusBadge("No Data Loaded", StatusBadge.State.NEUTRAL)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        layout.addWidget(self.lbl_status)
        
        return frame

    def _create_control_panel(self) -> QWidget:
        """Left side: File inputs and Load button."""
        panel = QFrame()
        panel.setFixedWidth(360)
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SURFACE};
                border-right: 1px solid {Colors.BORDER};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Section Title
        lbl = QLabel("SOURCE FILES")
        lbl.setStyleSheet(f"""
            font-weight: 700;
            color: {Colors.TEXT_SECONDARY};
            font-size: 11px;
            letter-spacing: 0.5px;
        """)
        layout.addWidget(lbl)
        
        # File Input Cards
        file_types = [
            ("collar", "Collar Data", "📍", True),
            ("survey", "Survey Data", "🧭", False),
            ("assay", "Assay Data", "🧪", False),
            ("lithology", "Lithology Data", "🪨", False),
        ]
        
        for ftype, label, icon, required in file_types:
            card = FileInputCard(
                label=label,
                file_filter="CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)",
                required=required,
                icon=icon
            )
            card.fileSelected.connect(lambda p, t=ftype: self._on_file_selected(t, p))
            card.fileCleared.connect(lambda t=ftype: self._on_file_cleared(t))
            self.file_cards[ftype] = card
            layout.addWidget(card)

        layout.addStretch()
        
        # Load Button
        self.btn_load = ActionButton("Load Data", variant="primary", icon="📥")
        self.btn_load.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.btn_load)
        
        return panel

    def _on_file_selected(self, file_type: str, path: str):
        """Handle file selection from card."""
        self.file_cards[file_type].set_file(path)

    def _on_file_cleared(self, file_type: str):
        """Handle file clear from card."""
        pass

    def _create_preview_panel(self) -> QWidget:
        """Right side: Tabbed tables to preview loaded data."""
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {Colors.BG_PRIMARY};")
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)
        
        # Preview header
        preview_header = QHBoxLayout()
        preview_title = QLabel("Data Preview")
        preview_title.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {Colors.TEXT_PRIMARY};
        """)
        preview_header.addWidget(preview_title)
        
        preview_hint = QLabel("First 50 rows shown")
        preview_hint.setStyleSheet(f"""
            font-size: 11px;
            color: {Colors.TEXT_MUTED};
        """)
        preview_header.addStretch()
        preview_header.addWidget(preview_hint)
        
        layout.addLayout(preview_header)
        
        # Modern tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: none;
                background-color: transparent;
            }}
            QTabBar {{
                background-color: transparent;
            }}
            QTabBar::tab {{
                background-color: {Colors.BG_SURFACE};
                color: {Colors.TEXT_SECONDARY};
                padding: 10px 20px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                border: 1px solid {Colors.BORDER};
                border-bottom: none;
                font-weight: 500;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.BG_SURFACE};
                color: {Colors.PRIMARY};
                border-bottom: 2px solid {Colors.PRIMARY};
                font-weight: 600;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.BG_HOVER};
            }}
        """)
        
        # Create Tables with icons
        self.table_collars = self._create_table()
        self.table_surveys = self._create_table()
        self.table_assays = self._create_table()
        self.table_litho = self._create_table()
        
        self.tabs.addTab(self.table_collars, "📍 Collars")
        self.tabs.addTab(self.table_surveys, "🧭 Surveys")
        self.tabs.addTab(self.table_assays, "🧪 Assays")
        self.tabs.addTab(self.table_litho, "🪨 Lithology")
        
        layout.addWidget(self.tabs)
        return panel

    def _create_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(0)
        table.setRowCount(0)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: {Colors.BG_PRIMARY};
                background-color: {Colors.BG_SURFACE};
            }}
        """)
        table.setShowGrid(True)
        return table

    def _create_footer(self) -> QFrame:
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet(f"""
            QFrame {{
                background-color: {Colors.BG_SURFACE};
                border-top: 1px solid {Colors.BORDER};
            }}
        """)
        
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(20, 0, 20, 0)
        
        self.msg_label = QLabel("Ready")
        self.msg_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-size: 11px;
        """)
        
        self.prog_bar = QProgressBar()
        self.prog_bar.setFixedSize(150, 6)
        self.prog_bar.setTextVisible(False)
        self.prog_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {Colors.BG_PRIMARY};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        self.prog_bar.hide()
        
        layout.addWidget(self.msg_label)
        layout.addStretch()
        layout.addWidget(self.prog_bar)
        
        return footer

    # --- LOGIC ---

    def _on_load_clicked(self):
        # Gather paths from cards
        paths = {}
        for k, card in self.file_cards.items():
            paths[k] = card.get_file_path()
            
        if not paths.get('collar'):
            QMessageBox.warning(self, "Missing File", "Collar file is required.")
            return

        # UI State -> Loading
        self._set_loading(True, "Reading files...")
        
        config = {
            "collar_file": paths.get('collar'),
            "survey_file": paths.get('survey'),
            "assay_file": paths.get('assay'),
            "lithology_file": paths.get('lithology')
        }
        
        # Call Controller
        if self.controller:
            self.controller.load_drillholes(config, self._on_load_complete)
        else:
            self._set_loading(False, "Error: Controller disconnected.")

    def _on_load_complete(self, result: Dict[str, Any]):
        self._set_loading(False)
        
        if "database" in result and result["database"]:
            self.database = result["database"]
            db = self.database
            
            # Update Status Badge
            count = len(db.collars) if not db.collars.empty else 0
            self.lbl_status.setText(f"Loaded: {count} Holes")
            self.lbl_status.set_state(StatusBadge.State.SUCCESS)
            
            self.msg_label.setText("Import successful.")
            
            # POPULATE PREVIEW TABLES (Top 50 rows)
            self._fill_table(self.table_collars, db.collars, ["hole_id", "x", "y", "z", "azimuth", "dip", "length"])
            self._fill_table(self.table_surveys, db.surveys, ["hole_id", "depth_from", "depth_to", "azimuth", "dip"])
            self._fill_table(self.table_assays, db.assays, ["hole_id", "depth_from", "depth_to", "values"])
            self._fill_table(self.table_litho, db.lithology, ["hole_id", "depth_from", "depth_to", "lith_code"])
            
        else:
            error_msg = result.get("error", "Failed to load database.")
            QMessageBox.critical(self, "Error", f"Import failed: {error_msg}")
            self.msg_label.setText("Import failed.")
            self.lbl_status.setText("Load Failed")
            self.lbl_status.set_state(StatusBadge.State.ERROR)

    def _fill_table(self, table: QTableWidget, data_list: List[Any], columns: List[str]):
        """Populates a table with the first 50 records."""
        if not data_list:
            table.setRowCount(0)
            table.setColumnCount(0)
            return
        
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels([c.replace("_", " ").title() for c in columns])
        
        # Limit to 50 rows for preview speed
        preview_data = data_list[:50]
        table.setRowCount(len(preview_data))
        
        for r, item in enumerate(preview_data):
            for c, col_name in enumerate(columns):
                val = getattr(item, col_name, "")
                
                # Special handling for Assay 'values' dict
                if col_name == "values" and isinstance(val, dict):
                    if val:
                        grade_summary = ", ".join([f"{k}: {v:.2f}" for k, v in list(val.items())[:3]])
                        if len(val) > 3:
                            grade_summary += f" ... (+{len(val) - 3} more)"
                        val = grade_summary
                    else:
                        val = ""
                
                if val is None:
                    val = ""
                
                item_widget = QTableWidgetItem(str(val))
                table.setItem(r, c, item_widget)
        
        # Resize columns to content
        table.resizeColumnsToContents()
        
        # Show row count in tab if truncated
        if len(data_list) > 50:
            # Update the corresponding tab text
            tab_idx = [self.table_collars, self.table_surveys, self.table_assays, self.table_litho].index(table)
            tab_names = ["📍 Collars", "🧭 Surveys", "🧪 Assays", "🪨 Lithology"]
            self.tabs.setTabText(tab_idx, f"{tab_names[tab_idx]} ({len(data_list):,})")

    def _set_loading(self, loading: bool, msg: str = ""):
        if loading:
            self.setCursor(Qt.CursorShape.WaitCursor)
            self.btn_load.setEnabled(False)
            self.prog_bar.setRange(0, 0)  # Indeterminate progress
            self.prog_bar.show()
            self.msg_label.setText(msg)
            self.lbl_status.setText("Loading...")
            self.lbl_status.set_state(StatusBadge.State.INFO)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.btn_load.setEnabled(True)
            self.prog_bar.hide()
            self.msg_label.setText(msg if msg else "Ready")
