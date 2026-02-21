"""
Drillhole Control Panel

Modern control panel for drillhole visualization settings.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Set

import logging

import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QScrollArea, QGridLayout,
    QLineEdit, QFrame, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QFont, QCursor

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea
from .signals import UISignals
from .modern_widgets import Colors, ActionButton, StatusBadge
from .modern_styles import get_theme_colors
from ..controllers.app_state import AppState, get_empty_state_message

logger = logging.getLogger(__name__)


class ModernSlider(QFrame):
    """A modern slider with label and value display.
    
    Features debounced value emission to prevent expensive operations
    from being triggered on every slider tick during dragging.
    """
    
    valueChanged = pyqtSignal(float)
    
    def __init__(
        self,
        label: str,
        min_val: float = 0.1,
        max_val: float = 5.0,
        default: float = 1.0,
        suffix: str = "m",
        decimals: int = 1,
        debounce_ms: int = 200,  # Debounce delay in milliseconds
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.suffix = suffix
        self.decimals = decimals
        self._multiplier = 10 ** decimals
        
        # Debounce timer to avoid expensive updates on every slider tick
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._emit_debounced_value)
        self._pending_value: Optional[float] = None
        
        self._setup_ui(label, default)
    
    def _setup_ui(self, label: str, default: float):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        self.label = QLabel(label)
        self.label.setFixedWidth(60)
        self.label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        layout.addWidget(self.label)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(self.min_val * self._multiplier))
        self.slider.setMaximum(int(self.max_val * self._multiplier))
        self.slider.setValue(int(default * self._multiplier))
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {Colors.BG_PRIMARY};
                height: 6px;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {Colors.PRIMARY};
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {Colors.PRIMARY_HOVER};
            }}
            QSlider::sub-page:horizontal {{
                background: {Colors.PRIMARY};
                border-radius: 3px;
            }}
        """)
        self.slider.valueChanged.connect(self._on_value_changed)
        layout.addWidget(self.slider, 1)
        
        self.value_label = QLabel(f"{default:.{self.decimals}f}{self.suffix}")
        self.value_label.setFixedWidth(50)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setStyleSheet(f"""
            color: {Colors.TEXT_PRIMARY};
            font-size: 11px;
            font-weight: 600;
            background-color: {Colors.BG_PRIMARY};
            padding: 4px 8px;
            border-radius: 4px;
        """)
        layout.addWidget(self.value_label)
    
    def _on_value_changed(self, int_val: int):
        """Handle slider value change - update label immediately, debounce signal emission."""
        val = int_val / self._multiplier
        # Update label immediately for visual feedback
        self.value_label.setText(f"{val:.{self.decimals}f}{self.suffix}")
        # Debounce the expensive signal emission
        self._pending_value = val
        self._debounce_timer.start()  # Restart timer on each change
    
    def _emit_debounced_value(self):
        """Emit the valueChanged signal after debounce delay."""
        if self._pending_value is not None:
            self.valueChanged.emit(self._pending_value)
            self._pending_value = None
    
    def value(self) -> float:
        return self.slider.value() / self._multiplier
    
    def setValue(self, val: float):
        self.slider.setValue(int(val * self._multiplier))


class ModernComboBox(QComboBox):
    """A modern styled combo box."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._apply_styles()

    def _apply_styles(self):
        """Apply theme-aware styles."""
        colors = get_theme_colors()
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: {colors.ELEVATED_BG};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                padding-right: 30px;
                color: {colors.TEXT_PRIMARY};
                font-size: 11px;
                min-height: 20px;
            }}
            QComboBox:hover {{
                border-color: {Colors.PRIMARY};
            }}
            QComboBox:focus {{
                border-color: {Colors.PRIMARY};
                outline: none;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {colors.TEXT_SECONDARY};
                margin-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors.ELEVATED_BG};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                selection-background-color: {Colors.PRIMARY};
                selection-color: {colors.TEXT_PRIMARY};
                padding: 4px;
            }}
        """)


class DrillholeControlPanel(BaseDockPanel):
    # Legacy signals - DEPRECATED (DR-005 fix)
    # All new code should use UISignals bus via self.signals
    # These are kept only for backward compatibility with external code
    plot_drillholes = pyqtSignal(str)
    clear_drillholes = pyqtSignal()
    radius_changed = pyqtSignal(float)
    color_mode_changed = pyqtSignal(str)
    assay_field_changed = pyqtSignal(str)
    show_ids_toggled = pyqtSignal(bool)
    hole_visibility_changed = pyqtSignal(str, bool)
    focus_selected_requested = pyqtSignal()

    # PanelManager metadata
    PANEL_ID = "DrillholeControlPanel"
    PANEL_NAME = "Drillhole Control Panel"
    PANEL_CATEGORY = PanelCategory.DRILLHOLE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.LEFT

    def __init__(self, parent: Optional[QWidget] = None, signals: Optional[UISignals] = None):
        QWidget.__init__(self, parent)
        
        self._hole_ids: List[str] = []
        self._hole_checkboxes: Dict[str, QCheckBox] = {}
        
        # Lithology filter state
        self._unique_liths: List[str] = []
        self._lith_checkboxes: Dict[str, QCheckBox] = {}
        self._lith_filter_debounce_timer: Optional[QTimer] = None
        
        # DataFrame references for UI access (DR-006 note)
        # These are references to data in DataRegistry, NOT copies.
        # They are set via drillholeDataLoaded signal for fast UI access.
        # For authoritative data, always query DataRegistry.get_drillhole_data().
        self._collars_df: Optional[pd.DataFrame] = None
        self._assays_df: Optional[pd.DataFrame] = None
        self._composites_df: Optional[pd.DataFrame] = None

        self.signals: Optional[UISignals] = signals
        self.renderer = None
        
        # Application state tracking
        self._app_state: AppState = AppState.EMPTY

        # Apply theme styles
        self._apply_theme_styles()
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(16)
        
        BaseDockPanel.__init__(self, parent)
        
        # Connect to registry
        registry = self.get_registry()
        if registry:
            try:
                registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
                if registry.get_drillhole_data():
                    self._on_drillhole_data_loaded(registry.get_drillhole_data())
            except Exception as e:
                logger.error(f"Failed to connect drillhole data signal: {e}", exc_info=True)

    def _apply_theme_styles(self) -> None:
        """Apply theme-aware styles to the panel."""
        colors = get_theme_colors()
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {colors.PANEL_BG};
                color: {colors.TEXT_PRIMARY};
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 12px;
            }}
            QGroupBox {{
                background-color: {colors.CARD_BG};
                border: 1px solid {colors.BORDER};
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 14px;
                font-weight: 600;
                font-size: 11px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {colors.TEXT_SECONDARY};
            }}
            QScrollArea {{
                border: 1px solid {colors.BORDER};
                background-color: {colors.CARD_BG};
                border-radius: 6px;
            }}
            QCheckBox {{
                spacing: 8px;
                color: {colors.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QCheckBox:checked {{
                color: {colors.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QCheckBox:hover {{
                color: {colors.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {colors.BORDER};
                background-color: {colors.ELEVATED_BG};
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.PRIMARY};
                border-color: {Colors.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Colors.PRIMARY};
            }}
            QLineEdit {{
                background-color: {colors.ELEVATED_BG};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                color: {colors.TEXT_PRIMARY};
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border-color: {Colors.PRIMARY};
                outline: none;
            }}
            QLineEdit::placeholder {{
                color: {colors.TEXT_HINT};
            }}
        """)

    def refresh_theme(self) -> None:
        """Refresh styles when theme changes."""
        self._apply_theme_styles()
        # Also update child widgets that have their own styles
        if hasattr(self, 'lith_container'):
            colors = get_theme_colors()
            self.lith_container.setStyleSheet(f"background-color: {colors.CARD_BG};")
        if hasattr(self, 'holes_container'):
            colors = get_theme_colors()
            self.holes_container.setStyleSheet(f"background-color: {colors.CARD_BG};")

    def setup_ui(self):
        self._init_ui()
        # Apply initial empty state
        self._apply_empty_state()

    def _init_ui(self):
        # Create main scroll area for the entire panel content
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        colors = get_theme_colors()
        main_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: {colors.CARD_BG};
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors.BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {Colors.PRIMARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        # Content widget inside scroll area
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 8, 0)  # Right margin for scrollbar
        content_layout.setSpacing(16)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("🔬 Drillhole Explorer")
        colors = get_theme_colors()
        title.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 700;
            color: {colors.TEXT_PRIMARY};
        """)
        header.addWidget(title)
        header.addStretch()
        
        self.status_badge = StatusBadge("No Data", StatusBadge.State.NEUTRAL)
        header.addWidget(self.status_badge)
        
        content_layout.addLayout(header)

        # --- VISUALIZATION SETTINGS ---
        gb_viz = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(gb_viz)
        viz_layout.setSpacing(12)
        viz_layout.setContentsMargins(12, 16, 12, 12)
        
        # Radius slider
        self.radius_slider = ModernSlider(
            "Radius",
            min_val=0.1,
            max_val=5.0,
            default=1.0,
            suffix="m"
        )
        self.radius_slider.valueChanged.connect(self._on_radius_changed)
        viz_layout.addWidget(self.radius_slider)

        # Color Mode
        color_row = QHBoxLayout()
        color_label = QLabel("Color By")
        color_label.setFixedWidth(60)
        color_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        self.color_mode_combo = ModernComboBox()
        self.color_mode_combo.addItems(["Lithology", "Assay"])
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_changed)
        color_row.addWidget(color_label)
        color_row.addWidget(self.color_mode_combo, 1)
        viz_layout.addLayout(color_row)

        # Assay Field
        element_row = QHBoxLayout()
        element_label = QLabel("Element")
        element_label.setFixedWidth(60)
        element_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        self.assay_field_combo = ModernComboBox()
        self.assay_field_combo.currentTextChanged.connect(self._on_assay_field_changed)
        self.assay_field_combo.setEnabled(False)
        element_row.addWidget(element_label)
        element_row.addWidget(self.assay_field_combo, 1)
        viz_layout.addLayout(element_row)

        # Dataset source
        source_row = QHBoxLayout()
        source_label = QLabel("Source")
        source_label.setFixedWidth(60)
        source_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")
        self.dataset_combo = ModernComboBox()
        self.dataset_combo.addItems(["Raw Assays"])
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        source_row.addWidget(source_label)
        source_row.addWidget(self.dataset_combo, 1)
        viz_layout.addLayout(source_row)

        content_layout.addWidget(gb_viz)

        # --- LITHOLOGY FILTER ---
        gb_lith = QGroupBox("Lithology Filter")
        lith_layout = QVBoxLayout(gb_lith)
        lith_layout.setSpacing(8)
        lith_layout.setContentsMargins(12, 16, 12, 12)
        
        # Lith filter buttons row
        lith_btn_row = QHBoxLayout()
        lith_btn_row.setSpacing(8)
        
        btn_lith_all = QPushButton("All")
        btn_lith_all.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        colors = get_theme_colors()
        btn_lith_all.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.ELEVATED_BG};
                color: {colors.TEXT_PRIMARY};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 10px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
                border-color: {Colors.PRIMARY};
            }}
        """)
        btn_lith_all.clicked.connect(self._select_all_liths)
        
        btn_lith_none = QPushButton("None")
        btn_lith_none.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_lith_none.setStyleSheet(btn_lith_all.styleSheet())
        btn_lith_none.clicked.connect(self._select_no_liths)
        
        lith_btn_row.addWidget(btn_lith_all)
        lith_btn_row.addWidget(btn_lith_none)
        lith_btn_row.addStretch()
        lith_layout.addLayout(lith_btn_row)
        
        # Scrollable lithology checkbox list
        lith_scroll = QScrollArea()
        lith_scroll.setWidgetResizable(True)
        lith_scroll.setMinimumHeight(80)
        lith_scroll.setMaximumHeight(150)
        
        self.lith_container = QWidget()
        colors = get_theme_colors()
        self.lith_container.setStyleSheet(f"background-color: {colors.CARD_BG};")
        self.lith_layout = QVBoxLayout(self.lith_container)
        self.lith_layout.setSpacing(4)
        self.lith_layout.setContentsMargins(8, 8, 8, 8)
        self.lith_layout.addStretch()
        lith_scroll.setWidget(self.lith_container)
        
        lith_layout.addWidget(lith_scroll)
        
        # Status label for lithology filter
        self.lith_status_label = QLabel("Load data to see lithologies")
        self.lith_status_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        self.lith_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lith_layout.addWidget(self.lith_status_label)
        
        content_layout.addWidget(gb_lith)

        # --- SELECTION ---
        gb_sel = QGroupBox("Hole Selection")
        sel_layout = QVBoxLayout(gb_sel)
        sel_layout.setSpacing(10)
        sel_layout.setContentsMargins(12, 16, 12, 12)
        
        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("🔍 Filter holes...")
        self.search_bar.textChanged.connect(self._filter_holes)
        sel_layout.addWidget(self.search_bar)

        # Select buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        
        btn_all = QPushButton("Select All")
        btn_all.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        colors = get_theme_colors()
        btn_all.setStyleSheet(f"""
            QPushButton {{
                background-color: {colors.ELEVATED_BG};
                color: {colors.TEXT_PRIMARY};
                border: 1px solid {colors.BORDER};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {colors.CARD_HOVER};
                border-color: {Colors.PRIMARY};
            }}
        """)
        btn_all.clicked.connect(self._select_all_holes)
        
        btn_none = QPushButton("Deselect All")
        btn_none.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_none.setStyleSheet(btn_all.styleSheet())
        btn_none.clicked.connect(self._select_no_holes)
        
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        sel_layout.addLayout(btn_row)
        
        # Show labels checkbox
        self.show_ids_checkbox = QCheckBox("Show Hole Labels")
        self.show_ids_checkbox.toggled.connect(self._on_show_ids_toggled)
        sel_layout.addWidget(self.show_ids_checkbox)

        # Scrollable hole list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        scroll.setMaximumHeight(200)
        
        self.holes_container = QWidget()
        colors = get_theme_colors()
        self.holes_container.setStyleSheet(f"background-color: {colors.CARD_BG};")
        self.holes_layout = QVBoxLayout(self.holes_container)
        self.holes_layout.setSpacing(4)
        self.holes_layout.setContentsMargins(8, 8, 8, 8)
        self.holes_layout.addStretch()
        scroll.setWidget(self.holes_container)
        
        sel_layout.addWidget(scroll)
        content_layout.addWidget(gb_sel)

        # --- ACTION BUTTONS ---
        content_layout.addStretch()
        
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)
        
        self.plot_button = ActionButton("Plot 3D", variant="primary", icon="🎯")
        self.plot_button.clicked.connect(self._on_plot_clicked)

        self.clear_button = ActionButton("Clear", variant="secondary")
        self.clear_button.clicked.connect(self._on_clear_clicked)

        action_layout.addWidget(self.plot_button)
        action_layout.addWidget(self.clear_button)
        content_layout.addLayout(action_layout)
        
        # Status label
        self.status_label = QLabel("Load drillhole data to begin")
        self.status_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 10px;
            padding: 8px 0;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.status_label)
        
        # Set content widget in scroll area
        main_scroll.setWidget(scroll_content)
        
        # Add scroll area to main layout
        self.main_layout.addWidget(main_scroll)

    # --- LOGIC ---

    def _on_drillhole_data_loaded(self, data: Dict):
        """Fast population using Pandas - deferred to prevent UI freeze."""
        logger.info(f"🔵 DrillholeControlPanel._on_drillhole_data_loaded called with data type: {type(data)}")
        logger.info(f"🔵 Data keys: {data.keys() if isinstance(data, dict) else 'NOT A DICT'}")

        if not isinstance(data, dict):
            logger.error(f"❌ Data is not a dict! Type: {type(data)}")
            return

        def _do_load():
            try:
                logger.info("🔵 _do_load() started")
                self._collars_df = data.get('collars')
                self._assays_df = data.get('assays')
                # AUDIT FIX: Can't use 'or' with DataFrames - use proper None check
                composites = data.get('composites')
                if composites is None:
                    composites = data.get('composites_df')
                self._composites_df = composites

                logger.info(f"🔵 Loaded DataFrames:")
                logger.info(f"   - Collars: {len(self._collars_df) if self._collars_df is not None else 'None'}")
                logger.info(f"   - Assays: {len(self._assays_df) if self._assays_df is not None else 'None'}")
                logger.info(f"   - Composites: {len(self._composites_df) if self._composites_df is not None else 'None'}")
                
                # Update Dataset Combo
                current_selection = self.dataset_combo.currentText() if self.dataset_combo.count() > 0 else None
                self.dataset_combo.blockSignals(True)
                self.dataset_combo.clear()
                available_datasets = []
                
                if self._assays_df is not None and not self._assays_df.empty:
                    self.dataset_combo.addItem("Raw Assays")
                    available_datasets.append("Raw Assays")
                if self._composites_df is not None and not self._composites_df.empty:
                    self.dataset_combo.addItem("Composites")
                    available_datasets.append("Composites")
                
                if current_selection in available_datasets:
                    self.dataset_combo.setCurrentText(current_selection)
                elif available_datasets:
                    self.dataset_combo.setCurrentIndex(0)
                else:
                    self.dataset_combo.addItem("No Data")

                self.dataset_combo.blockSignals(False)

                # Populate Holes
                if self._collars_df is not None and not self._collars_df.empty:
                    hole_col = next((c for c in self._collars_df.columns if c.lower() in ['holeid','hole_id','bhid']), None)
                    if hole_col:
                        self._hole_ids = sorted(self._collars_df[hole_col].astype(str).unique().tolist())
                        self._rebuild_checklist()
                        self.status_label.setText(f"{len(self._hole_ids)} holes available")
                        self.status_badge.setText(f"{len(self._hole_ids)} Holes")
                        self.status_badge.set_state(StatusBadge.State.SUCCESS)

                self._populate_assay_fields()
                
                # Extract unique lithology codes for filtering
                # We need to scan ALL possible sources of lithology data to match what gets rendered
                self._unique_liths = []
                unique_liths_set = set()
                
                # Source 1: Dedicated lithology DataFrame
                lith_df = data.get('lithology')
                if lith_df is not None and not lith_df.empty:
                    # Find the lithology code column
                    lith_col = None
                    for col in lith_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            lith_col = col
                            break
                    
                    if lith_col:
                        unique_liths_set.update(
                            lith_df[lith_col].dropna().astype(str).unique().tolist()
                        )
                
                # Source 2: Lithology column in assays DataFrame (some datasets embed lithology in assays)
                assays_df = data.get('assays')
                if assays_df is not None and not assays_df.empty:
                    for col in assays_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            unique_liths_set.update(
                                assays_df[col].dropna().astype(str).unique().tolist()
                            )
                            break  # Only check first matching column
                
                # Source 3: Lithology column in composites DataFrame
                composites_df = data.get('composites')
                if composites_df is None:
                    composites_df = data.get('composites_df')
                if composites_df is not None and not composites_df.empty:
                    for col in composites_df.columns:
                        if col.lower() in ['lith_code', 'lithology', 'code', 'lith']:
                            unique_liths_set.update(
                                composites_df[col].dropna().astype(str).unique().tolist()
                            )
                            break  # Only check first matching column
                
                # Convert to sorted list and filter out empty strings
                self._unique_liths = sorted([lith for lith in unique_liths_set if lith and str(lith).strip()])
                
                self._rebuild_lith_checklist()
                
                # Auto-select color mode based on available data:
                # - If no lithology data but assay data exists, default to "Assay"
                # - If lithology data exists, keep "Lithology" as default
                has_lith_data = len(self._unique_liths) > 0
                has_assay_data = self.assay_field_combo.count() > 0
                
                if not has_lith_data and has_assay_data:
                    # No lithology data available - switch to Assay mode
                    self.color_mode_combo.blockSignals(True)
                    self.color_mode_combo.setCurrentText("Assay")
                    self.color_mode_combo.blockSignals(False)
                    self.assay_field_combo.setEnabled(True)
                    logger.info("Auto-selected Assay color mode (no lithology data found)")
                elif has_lith_data:
                    # Lithology data available - ensure Lithology mode
                    self.color_mode_combo.blockSignals(True)
                    self.color_mode_combo.setCurrentText("Lithology")
                    self.color_mode_combo.blockSignals(False)
                    self.assay_field_combo.setEnabled(False)
                    logger.info(f"Auto-selected Lithology color mode ({len(self._unique_liths)} codes found)")

                logger.info("🔵 _do_load() COMPLETED SUCCESSFULLY")
                logger.info(f"🔵 Final state:")
                logger.info(f"   - _hole_ids: {len(self._hole_ids) if self._hole_ids else 0} holes")
                logger.info(f"   - dataset_combo: {self.dataset_combo.count()} items")
                logger.info(f"   - status_label: '{self.status_label.text()}'")
                logger.info(f"   - Panel visible: {self.isVisible()}")
                logger.info(f"   - Panel enabled: {self.isEnabled()}")

            except Exception as e:
                logger.error(f"❌ Error loading drillhole data: {e}", exc_info=True)
        
        QTimer.singleShot(10, _do_load)

    def _rebuild_checklist(self):
        """Create checkboxes for holes (limited to 100)."""
        for i in reversed(range(self.holes_layout.count())):
            w = self.holes_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
            
        self._hole_checkboxes.clear()
        self.holes_layout.addStretch()

        display_limit = 100
        
        for hid in self._hole_ids[:display_limit]:
            cb = QCheckBox(hid)
            cb.setChecked(True)
            cb.toggled.connect(lambda c, h=hid: self._on_hole_visibility_changed(h, c))
            self.holes_layout.insertWidget(self.holes_layout.count()-1, cb)
            self._hole_checkboxes[hid] = cb
            
        if len(self._hole_ids) > display_limit:
            lbl = QLabel(f"... +{len(self._hole_ids) - display_limit} more (use search)")
            lbl.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px; font-style: italic;")
            self.holes_layout.insertWidget(self.holes_layout.count()-1, lbl)

    def _populate_assay_fields(self):
        """Populate assay fields based on selected dataset."""
        dataset_name = self.dataset_combo.currentText()
        df = self._composites_df if dataset_name == "Composites" else self._assays_df
        
        if df is None or df.empty:
            self.assay_field_combo.clear()
            return
        
        ignore = {
            'holeid','hole_id','from','to','x','y','z','length','depth_from','depth_to','global_interval_id',
            'sample_count','total_mass','total_length','support','is_partial',
            'method','weighting','element_weights','merged_partial','merged_partial_auto'
        }
        cols = [c for c in df.columns if c.lower() not in ignore and pd.api.types.is_numeric_dtype(df[c])]
        
        self.assay_field_combo.blockSignals(True)
        self.assay_field_combo.clear()
        self.assay_field_combo.addItems(sorted(cols))
        self.assay_field_combo.blockSignals(False)

    def _filter_holes(self, text):
        text = text.lower()
        for hid, cb in self._hole_checkboxes.items():
            cb.setVisible(text in hid.lower())

    # --- Lithology Filter Methods ---
    
    def _select_all_liths(self):
        """Select all lithology types."""
        # Block signals during bulk update to prevent multiple re-renders
        for cb in self._lith_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        # Emit signal once after all changes
        self._emit_lith_filter_signal()
    
    def _select_no_liths(self):
        """Deselect all lithology types."""
        # Block signals during bulk update to prevent multiple re-renders
        for cb in self._lith_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        # Emit signal once after all changes
        self._emit_lith_filter_signal()
    
    def _on_lith_visibility_changed(self, lith_code: str, visible: bool):
        """Handle lithology checkbox toggle - debounced signal emission.
        
        Uses a debounce timer to avoid triggering re-renders on every checkbox click.
        This allows users to select/deselect multiple lithologies quickly before
        the expensive re-render kicks in.
        """
        # Create debounce timer if needed
        if self._lith_filter_debounce_timer is None:
            self._lith_filter_debounce_timer = QTimer(self)
            self._lith_filter_debounce_timer.setSingleShot(True)
            self._lith_filter_debounce_timer.setInterval(500)  # 500ms debounce
            self._lith_filter_debounce_timer.timeout.connect(self._emit_lith_filter_signal)
        
        # Restart the timer on each checkbox change
        self._lith_filter_debounce_timer.start()
    
    def _emit_lith_filter_signal(self):
        """Emit the lithology filter signal after debounce delay."""
        selected = self.get_selected_lithologies()
        if self.signals:
            self.signals.drillholeLithFilterChanged.emit(selected)
    
    def _rebuild_lith_checklist(self):
        """Create checkboxes for unique lithology codes."""
        # Clear existing checkboxes
        for i in reversed(range(self.lith_layout.count())):
            w = self.lith_layout.itemAt(i).widget()
            if w:
                w.deleteLater()
        
        self._lith_checkboxes.clear()
        self.lith_layout.addStretch()
        
        if not self._unique_liths:
            self.lith_status_label.setText("No lithology data available")
            return
        
        for lith in self._unique_liths:
            cb = QCheckBox(lith)
            cb.blockSignals(True)  # Prevent signal during initial setup
            cb.setChecked(True)  # All selected by default
            cb.blockSignals(False)
            cb.toggled.connect(lambda c, l=lith: self._on_lith_visibility_changed(l, c))
            self.lith_layout.insertWidget(self.lith_layout.count() - 1, cb)
            self._lith_checkboxes[lith] = cb
        
        self.lith_status_label.setText(f"{len(self._unique_liths)} lithology types")

    def _select_all_holes(self):
        for cb in self._hole_checkboxes.values():
            if cb.isVisible():
                cb.setChecked(True)
            
    def _select_no_holes(self):
        for cb in self._hole_checkboxes.values():
            if cb.isVisible():
                cb.setChecked(False)

    def _on_plot_clicked(self):
        ds = self.dataset_combo.currentText()
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholePlotRequested.emit(ds)
        else:
            # Fallback to legacy signal only if UISignals not available
            self.plot_drillholes.emit(ds)

    def _on_clear_clicked(self):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeClearRequested.emit()
        else:
            self.clear_drillholes.emit()

    def _on_radius_changed(self, radius: float):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeRadiusChanged.emit(radius)
        else:
            self.radius_changed.emit(radius)

    def _on_color_mode_changed(self, mode):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeColorModeChanged.emit(mode)
        else:
            self.color_mode_changed.emit(mode)
        self.assay_field_combo.setEnabled(mode == "Assay")

    def _on_dataset_changed(self, dataset_name: str):
        self._populate_assay_fields()
        # If drillholes are already plotted, re-plot with new dataset
        if self.renderer and "drillholes" in self.renderer.active_layers:
            if self.signals:
                self.signals.drillholePlotRequested.emit(dataset_name)
    
    def _on_assay_field_changed(self, field):
        if field:
            # Prefer UISignals bus (DR-005 fix)
            if self.signals:
                self.signals.drillholeAssayFieldChanged.emit(field)
            else:
                self.assay_field_changed.emit(field)

    def _on_show_ids_toggled(self, checked: bool):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeShowIdsToggled.emit(checked)
        else:
            self.show_ids_toggled.emit(checked)

    def _on_hole_visibility_changed(self, hole_id: str, visible: bool):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeVisibilityChanged.emit(hole_id, visible)
        else:
            self.hole_visibility_changed.emit(hole_id, visible)

    def _on_focus_requested(self):
        # Prefer UISignals bus (DR-005 fix)
        if self.signals:
            self.signals.drillholeFocusRequested.emit()
        else:
            self.focus_selected_requested.emit()

    # Public getters
    def get_visible_holes(self) -> Set[str]:
        """Get set of hole IDs that should be visible.
        
        Includes:
        - Holes with checkboxes that are checked (first 100)
        - Holes without checkboxes (beyond first 100) - always included
        """
        # Get checked holes from UI checkboxes
        checked_holes = {h for h, cb in self._hole_checkboxes.items() if cb.isChecked()}
        
        # Add holes beyond the checkbox limit (they don't have checkboxes, so default to visible)
        display_limit = 100
        if len(self._hole_ids) > display_limit:
            # Include all holes beyond the display limit
            holes_without_checkboxes = set(self._hole_ids[display_limit:])
            checked_holes = checked_holes.union(holes_without_checkboxes)
        
        return checked_holes

    def get_color_mode(self):
        return self.color_mode_combo.currentText()
    
    def get_assay_field(self):
        return self.assay_field_combo.currentText()
    
    def get_radius(self) -> float:
        return self.radius_slider.value()
    
    def get_dataset(self) -> str:
        return self.dataset_combo.currentText() if self.dataset_combo.count() > 0 else "Raw Assays"
    
    def get_selected_lithologies(self) -> List[str]:
        """Get list of selected lithology codes for filtering.
        
        Returns:
            List of selected lithology codes. 
            - Empty list when ALL are selected means "show all" (no filter)
            - Empty list when NONE are selected returns ["__NONE__"] to signal "show nothing"
        """
        if not self._lith_checkboxes:
            return []  # No filter data - show all
        
        selected = [lith for lith, cb in self._lith_checkboxes.items() if cb.isChecked()]
        
        # If all are selected, return empty list (meaning "no filter")
        if len(selected) == len(self._lith_checkboxes):
            return []
        
        # If NONE are selected, return special marker to indicate "show nothing"
        if len(selected) == 0:
            return ["__NONE__"]  # Special marker - caller should not render anything
        
        return selected
    
    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_renderer(self, renderer):
        self.renderer = renderer

    # =========================================================================
    # Application State Handling
    # =========================================================================
    
    def on_app_state_changed(self, state: int) -> None:
        """
        Handle application state changes.
        
        Controls are disabled in EMPTY state, show helpful message.
        
        Args:
            state: AppState enum value (as int for signal compatibility)
        """
        try:
            new_state = AppState(state)
        except ValueError:
            logger.warning(f"Invalid app state value: {state}")
            return
        
        if self._app_state == new_state:
            return
        
        old_state = self._app_state
        self._app_state = new_state
        logger.debug(f"DrillholeControlPanel: State changed {old_state.name} -> {new_state.name}")
        
        # Apply state-specific UI rules
        if new_state == AppState.EMPTY:
            self._apply_empty_state()
        elif new_state == AppState.DATA_LOADED:
            self._apply_data_loaded_state()
        elif new_state == AppState.RENDERED:
            self._apply_rendered_state()
        elif new_state == AppState.BUSY:
            self._apply_busy_state()
    
    def _apply_empty_state(self) -> None:
        """Apply EMPTY state: Disable all controls, show helpful message."""
        # Disable controls
        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(False)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(False)
        if hasattr(self, 'assay_field_combo'):
            self.assay_field_combo.setEnabled(False)
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(False)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(False)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(False)
        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(False)
        if hasattr(self, 'focus_button'):
            self.focus_button.setEnabled(False)
        
        # Disable lithology filter checkboxes
        for cb in self._lith_checkboxes.values():
            cb.setEnabled(False)
        
        # Update status badge
        if hasattr(self, 'status_badge'):
            self.status_badge.setText("No Data")
            self.status_badge.set_state(StatusBadge.State.NEUTRAL)
        
        # Update status label with helpful message
        if hasattr(self, 'status_label'):
            self.status_label.setText(get_empty_state_message("drillhole_controls"))
    
    def _apply_data_loaded_state(self) -> None:
        """Apply DATA_LOADED state: Enable controls for configuring visualization before plotting.
        
        Users should be able to configure visualization settings (radius, color mode, etc.)
        BEFORE clicking Plot 3D. This allows them to set up how the data will be displayed.
        """
        # Enable data selection controls
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(True)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(True)
        
        # Enable visualization configuration controls so user can set up before plotting
        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(True)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(True)
        if hasattr(self, 'assay_field_combo'):
            # Only enable if color mode is "Assay"
            mode = self.color_mode_combo.currentText() if hasattr(self, 'color_mode_combo') else ""
            self.assay_field_combo.setEnabled(mode == "Assay")
        
        # Enable hole selection controls
        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(True)
        
        # Enable lithology filter checkboxes
        for cb in self._lith_checkboxes.values():
            cb.setEnabled(True)
        
        # Clear button enabled (user can clear even before plotting if needed)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(True)
    
    def _apply_rendered_state(self) -> None:
        """Apply RENDERED state: Enable all controls."""
        if hasattr(self, 'radius_slider'):
            self.radius_slider.setEnabled(True)
        if hasattr(self, 'color_mode_combo'):
            self.color_mode_combo.setEnabled(True)
        if hasattr(self, 'assay_field_combo'):
            # Only enable if color mode is "Assay"
            mode = self.color_mode_combo.currentText() if hasattr(self, 'color_mode_combo') else ""
            self.assay_field_combo.setEnabled(mode == "Assay")
        if hasattr(self, 'dataset_combo'):
            self.dataset_combo.setEnabled(True)
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(True)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(True)
        if hasattr(self, 'show_ids_checkbox'):
            self.show_ids_checkbox.setEnabled(True)
        if hasattr(self, 'focus_button'):
            self.focus_button.setEnabled(True)
        
        # Enable lithology filter checkboxes
        for cb in self._lith_checkboxes.values():
            cb.setEnabled(True)
    
    def _apply_busy_state(self) -> None:
        """Apply BUSY state: Disable interactive controls during processing."""
        if hasattr(self, 'plot_button'):
            self.plot_button.setEnabled(False)
        if hasattr(self, 'clear_button'):
            self.clear_button.setEnabled(False)
