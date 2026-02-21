"""
Geological Explorer Panel

Modern control panel for geological model visualization settings.
Controls rendering of LoopStructural-generated surfaces and solids in the main 3D viewer.

Features:
- View mode presets (Surfaces Only, Solids Only, etc.)
- Individual surface/solid layer toggles
- Wireframe support for solids
- Contact points with misfit error coloring
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
import logging

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from .modern_styles import get_theme_colors, ModernColors
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QScrollArea, QFrame, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QHeaderView
)
from PyQt6.QtGui import QCursor

from .base_panel import BaseDockPanel
from .panel_manager import PanelCategory, DockArea
from .signals import UISignals
from .modern_widgets import Colors, ActionButton, StatusBadge

logger = logging.getLogger(__name__)


class ModernSlider(QFrame):
    """A modern slider with label and value display."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float = 0.0,
        max_val: float = 1.0,
        default: float = 1.0,
        suffix: str = "",
        decimals: int = 2,
        debounce_ms: int = 200,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)

    def _get_stylesheet(self) -> str:
        """Get the stylesheet for current theme."""
        return f"""
        
                    QComboBox {{
                        background-color: #333333;
                        border: 1px solid #666666;
                        border-radius: 6px;
                        padding: 6px 12px;
                        padding-right: 30px;
                        color: {ModernColors.TEXT_PRIMARY};
                        font-size: 11px;
                        min-height: 20px;
                    }}
                    QComboBox:hover {{
                        border-color: {{Colors.PRIMARY}};
                    }}
                    QComboBox:focus {{
                        border-color: {{Colors.PRIMARY}};
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
                        border-top: 6px solid #cccccc;
                        margin-right: 8px;
                    }}
                    QComboBox QAbstractItemView {{
                        background-color: #333333;
                        border: 1px solid #666666;
                        border-radius: 6px;
                        selection-background-color: {{Colors.PRIMARY}};
                        selection-color: {ModernColors.TEXT_PRIMARY};
                        padding: 4px;
                    }}
                
        """

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            # Rebuild stylesheet with new theme colors
            self.setStyleSheet(self._get_stylesheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        self.min_val = min_val
        self.max_val = max_val
        self.suffix = suffix
        self.decimals = decimals
        self._multiplier = 10 ** decimals

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
        val = int_val / self._multiplier
        self.value_label.setText(f"{val:.{self.decimals}f}{self.suffix}")
        self._pending_value = val
        self._debounce_timer.start()

    def _emit_debounced_value(self):
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


class GeologicalExplorerPanel(BaseDockPanel):
    """
    Control panel for geological model visualization in the main 3D viewer.

    Features:
    - View mode presets (Surfaces Only, Solids Only, Contacts Only, etc.)
    - Individual surface layer toggles
    - Individual solid/domain layer toggles with wireframe option
    - Contact points display with misfit error coloring
    - Separate opacity controls for surfaces and solids
    """

    # PanelManager metadata
    PANEL_ID = "GeologicalExplorerPanel"
    PANEL_NAME = "Geological Explorer"
    PANEL_CATEGORY = PanelCategory.RESOURCE
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT
    PANEL_TOOLTIP = "Control geological model visualization"
    PANEL_ICON = "geology"

    def __init__(self, parent: Optional[QWidget] = None, signals: Optional[UISignals] = None):
        QWidget.__init__(self, parent)

        # Layer tracking
        self._surface_items: Dict[str, QTreeWidgetItem] = {}
        self._solid_items: Dict[str, QTreeWidgetItem] = {}
        self._surface_names: List[str] = []
        self._solid_names: List[str] = []

        # Geology data reference
        self._geology_package: Optional[Dict] = None
        self._contact_count: int = 0
        self._p90_error: float = 0.0
        self._mean_error: float = 0.0

        self.signals: Optional[UISignals] = signals
        self.renderer = None

        # High contrast dark theme
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #000000;
                color: {ModernColors.TEXT_PRIMARY};
                font-family: 'Segoe UI', -apple-system, sans-serif;
                font-size: 12px;
            }}
            QGroupBox {{
                background-color: #1a1a1a;
                border: 1px solid #444444;
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
                color: #cccccc;
            }}
            QScrollArea {{
                border: 1px solid #444444;
                background-color: #1a1a1a;
                border-radius: 6px;
            }}
            QCheckBox {{
                spacing: 8px;
                color: {ModernColors.TEXT_PRIMARY};
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #666666;
                background-color: #333333;
            }}
            QCheckBox::indicator:checked {{
                background-color: {{Colors.PRIMARY}};
                border-color: {{Colors.PRIMARY}};
            }}
            QCheckBox::indicator:hover {{
                border-color: {{Colors.PRIMARY}};
            }}
            QTreeWidget {{
                background-color: #1a1a1a;
                border: 1px solid #444444;
                border-radius: 6px;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 4px 2px;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QTreeWidget::item:hover {{
                background-color: #333333;
            }}
            QTreeWidget::item:selected {{
                background-color: #444444;
            }}
            QTreeWidget::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 2px solid #666666;
                background-color: #333333;
            }}
            QTreeWidget::indicator:checked {{
                background-color: {{Colors.PRIMARY}};
                border-color: {{Colors.PRIMARY}};
            }}
        """)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(16)

        BaseDockPanel.__init__(self, parent)

    def setup_ui(self):
        self._init_ui()
        self._apply_empty_state()

    def _init_ui(self):
        """Build the panel UI."""
        # Create main scroll area
        main_scroll = QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        main_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background-color: #1a1a1a;
                width: 8px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #444444;
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

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(scroll_content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(12)

        # Header
        header = QHBoxLayout()
        title = QLabel("Geological Explorer")
        title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {ModernColors.TEXT_PRIMARY};")
        header.addWidget(title)
        header.addStretch()

        self.status_badge = StatusBadge("No Data", StatusBadge.State.NEUTRAL)
        header.addWidget(self.status_badge)
        content_layout.addLayout(header)

        # --- VIEW MODE SECTION ---
        gb_view = QGroupBox("View Mode")
        view_layout = QVBoxLayout(gb_view)
        view_layout.setSpacing(8)
        view_layout.setContentsMargins(12, 16, 12, 12)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Preset")
        mode_label.setFixedWidth(50)
        mode_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")

        self.view_mode_combo = ModernComboBox()
        self.view_mode_combo.addItems([
            "Surfaces Only",
            "Solids Only",
            "Contacts Only",
            "Surfaces + Solids",
            "All"
        ])
        self.view_mode_combo.setCurrentIndex(3)  # Default: Surfaces + Solids (more useful for geology modeling)
        self.view_mode_combo.setToolTip(
            "Surfaces Only: Show geological surfaces\n"
            "Solids Only: Show solid domains\n"
            "Contacts Only: Show contact points with misfit coloring\n"
            "Surfaces + Solids: Show both\n"
            "All: Show everything"
        )
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.view_mode_combo, 1)
        view_layout.addLayout(mode_row)

        content_layout.addWidget(gb_view)

        # --- SURFACES SECTION ---
        gb_surfaces = QGroupBox("Surfaces")
        surf_layout = QVBoxLayout(gb_surfaces)
        surf_layout.setSpacing(8)
        surf_layout.setContentsMargins(12, 16, 12, 12)

        # All surfaces toggle
        surf_header = QHBoxLayout()
        self.all_surfaces_cb = QCheckBox("Show All")
        self.all_surfaces_cb.setChecked(True)
        self.all_surfaces_cb.toggled.connect(self._on_all_surfaces_toggled)
        surf_header.addWidget(self.all_surfaces_cb)
        surf_header.addStretch()
        self.surface_count_label = QLabel("0 surfaces")
        self.surface_count_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        surf_header.addWidget(self.surface_count_label)
        surf_layout.addLayout(surf_header)

        # Surface tree
        self.surface_tree = QTreeWidget()
        self.surface_tree.setHeaderHidden(True)
        self.surface_tree.setRootIsDecorated(False)
        self.surface_tree.setMinimumHeight(60)
        self.surface_tree.setMaximumHeight(120)
        self.surface_tree.itemChanged.connect(self._on_surface_item_changed)
        surf_layout.addWidget(self.surface_tree)

        # Surface opacity
        self.surface_opacity_slider = ModernSlider(
            "Opacity", min_val=0.1, max_val=1.0, default=1.0, decimals=2, debounce_ms=200
        )
        self.surface_opacity_slider.valueChanged.connect(self._on_surface_opacity_changed)
        surf_layout.addWidget(self.surface_opacity_slider)

        content_layout.addWidget(gb_surfaces)

        # --- SOLIDS SECTION ---
        gb_solids = QGroupBox("Solids / Domains")
        solid_layout = QVBoxLayout(gb_solids)
        solid_layout.setSpacing(8)
        solid_layout.setContentsMargins(12, 16, 12, 12)

        # All solids toggle
        solid_header = QHBoxLayout()
        self.all_solids_cb = QCheckBox("Show All")
        self.all_solids_cb.setChecked(True)
        self.all_solids_cb.toggled.connect(self._on_all_solids_toggled)
        solid_header.addWidget(self.all_solids_cb)
        solid_header.addStretch()
        self.solid_count_label = QLabel("0 domains")
        self.solid_count_label.setStyleSheet(f"color: {Colors.TEXT_MUTED}; font-size: 10px;")
        solid_header.addWidget(self.solid_count_label)
        solid_layout.addLayout(solid_header)

        # Solid tree
        self.solid_tree = QTreeWidget()
        self.solid_tree.setHeaderHidden(True)
        self.solid_tree.setRootIsDecorated(False)
        self.solid_tree.setMinimumHeight(60)
        self.solid_tree.setMaximumHeight(120)
        self.solid_tree.itemChanged.connect(self._on_solid_item_changed)
        solid_layout.addWidget(self.solid_tree)

        # Wireframe toggle
        self.wireframe_cb = QCheckBox("Show Wireframe")
        self.wireframe_cb.setChecked(False)
        self.wireframe_cb.toggled.connect(self._on_wireframe_toggled)
        solid_layout.addWidget(self.wireframe_cb)

        # Solid opacity
        self.solid_opacity_slider = ModernSlider(
            "Opacity", min_val=0.1, max_val=1.0, default=1.0, decimals=2, debounce_ms=200
        )
        self.solid_opacity_slider.valueChanged.connect(self._on_solid_opacity_changed)
        solid_layout.addWidget(self.solid_opacity_slider)

        content_layout.addWidget(gb_solids)

        # --- CONTACTS SECTION ---
        gb_contacts = QGroupBox("Contacts (Misfit Error)")
        contact_layout = QVBoxLayout(gb_contacts)
        contact_layout.setSpacing(8)
        contact_layout.setContentsMargins(12, 16, 12, 12)

        self.show_contacts_cb = QCheckBox("Show Contacts")
        self.show_contacts_cb.setChecked(False)  # Hidden by default per user preference
        self.show_contacts_cb.toggled.connect(self._on_contacts_toggled)
        contact_layout.addWidget(self.show_contacts_cb)

        # Contact stats
        self.contact_stats_label = QLabel("No contact data")
        self.contact_stats_label.setStyleSheet(f"""
            color: {Colors.TEXT_SECONDARY};
            font-size: 10px;
            padding: 4px 8px;
            background-color: #262626;
            border-radius: 4px;
        """)
        contact_layout.addWidget(self.contact_stats_label)

        content_layout.addWidget(gb_contacts)

        # --- APPEARANCE SECTION ---
        gb_appear = QGroupBox("Appearance")
        appear_layout = QVBoxLayout(gb_appear)
        appear_layout.setSpacing(8)
        appear_layout.setContentsMargins(12, 16, 12, 12)

        palette_row = QHBoxLayout()
        palette_label = QLabel("Palette")
        palette_label.setFixedWidth(50)
        palette_label.setStyleSheet(f"color: {Colors.TEXT_SECONDARY}; font-size: 11px;")

        self.palette_combo = ModernComboBox()
        self.palette_combo.addItems(["Geologic", "Viridis", "Tab10", "Spectral"])
        self.palette_combo.currentTextChanged.connect(self._on_palette_changed)
        palette_row.addWidget(palette_label)
        palette_row.addWidget(self.palette_combo, 1)
        appear_layout.addLayout(palette_row)

        content_layout.addWidget(gb_appear)

        # --- ACTION BUTTONS ---
        content_layout.addStretch()

        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.reset_view_btn = ActionButton("Reset View", variant="secondary")
        self.reset_view_btn.clicked.connect(self._on_reset_view)

        self.clear_btn = ActionButton("Clear All", variant="secondary")
        self.clear_btn.clicked.connect(self._on_clear)

        action_layout.addWidget(self.reset_view_btn)
        action_layout.addWidget(self.clear_btn)
        content_layout.addLayout(action_layout)

        # Status label
        self.status_label = QLabel("Build geological model to begin")
        self.status_label.setStyleSheet(f"""
            color: {Colors.TEXT_MUTED};
            font-size: 10px;
            padding: 8px 0;
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.status_label)

        main_scroll.setWidget(scroll_content)
        self.main_layout.addWidget(main_scroll)

    # --- SIGNAL HANDLERS ---

    def _on_view_mode_changed(self, mode_text: str):
        """Handle view mode preset change."""
        mode_map = {
            "Surfaces Only": "surfaces_only",
            "Solids Only": "solids_only",
            "Contacts Only": "contacts_only",
            "Surfaces + Solids": "surfaces_solids",
            "All": "all"
        }
        mode = mode_map.get(mode_text, "surfaces_only")

        if self.signals:
            self.signals.geologyViewModeChanged.emit(mode)

        # Update checkboxes to match view mode (without triggering signals)
        self._update_checkboxes_for_view_mode(mode)
        logger.info(f"Geology view mode changed to: {mode}")

    def _update_checkboxes_for_view_mode(self, mode: str):
        """Update checkbox states based on view mode."""
        modes = {
            "surfaces_only": (True, False, False),
            "solids_only": (False, True, False),
            "contacts_only": (False, False, True),
            "surfaces_solids": (True, True, False),
            "all": (True, True, True),
        }
        surfaces_vis, solids_vis, contacts_vis = modes.get(mode, (True, False, False))

        # Block signals while updating
        self.all_surfaces_cb.blockSignals(True)
        self.all_solids_cb.blockSignals(True)
        self.show_contacts_cb.blockSignals(True)

        self.all_surfaces_cb.setChecked(surfaces_vis)
        self.all_solids_cb.setChecked(solids_vis)
        self.show_contacts_cb.setChecked(contacts_vis)

        self.all_surfaces_cb.blockSignals(False)
        self.all_solids_cb.blockSignals(False)
        self.show_contacts_cb.blockSignals(False)

    def _on_all_surfaces_toggled(self, checked: bool):
        """Toggle all surfaces visibility."""
        # Update all surface tree items
        self.surface_tree.blockSignals(True)
        for i in range(self.surface_tree.topLevelItemCount()):
            item = self.surface_tree.topLevelItem(i)
            item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.surface_tree.blockSignals(False)

        # Emit signal for all surfaces
        if self.signals:
            self.signals.geologySurfacesVisibilityChanged.emit(checked)
        logger.debug(f"All surfaces visibility: {checked}")

    def _on_all_solids_toggled(self, checked: bool):
        """Toggle all solids visibility."""
        # Update all solid tree items
        self.solid_tree.blockSignals(True)
        for i in range(self.solid_tree.topLevelItemCount()):
            item = self.solid_tree.topLevelItem(i)
            item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.solid_tree.blockSignals(False)

        # Emit visibility for all solids
        for name in self._solid_names:
            layer_name = f"GeoSolid: {name}"
            if self.signals:
                self.signals.geologyLayerVisibilityChanged.emit(layer_name, checked)
        logger.debug(f"All solids visibility: {checked}")

    def _on_surface_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle individual surface checkbox change."""
        if column != 0:
            return
        surface_name = item.text(0)
        checked = item.checkState(0) == Qt.CheckState.Checked
        layer_name = f"GeoSurface: {surface_name}"

        if self.signals:
            self.signals.geologyLayerVisibilityChanged.emit(layer_name, checked)
        logger.debug(f"Surface '{surface_name}' visibility: {checked}")

    def _on_solid_item_changed(self, item: QTreeWidgetItem, column: int):
        """Handle individual solid checkbox change."""
        if column != 0:
            return
        solid_name = item.text(0)
        checked = item.checkState(0) == Qt.CheckState.Checked
        layer_name = f"GeoSolid: {solid_name}"

        if self.signals:
            self.signals.geologyLayerVisibilityChanged.emit(layer_name, checked)
        logger.debug(f"Solid '{solid_name}' visibility: {checked}")

    def _on_wireframe_toggled(self, checked: bool):
        """Handle wireframe toggle."""
        if self.signals:
            self.signals.geologyWireframeToggled.emit(checked)
        logger.debug(f"Wireframe visibility: {checked}")

    def _on_contacts_toggled(self, checked: bool):
        """Handle contacts visibility toggle."""
        if self.signals:
            self.signals.geologyContactsVisibilityChanged.emit(checked)
        logger.debug(f"Contacts visibility: {checked}")

    def _on_surface_opacity_changed(self, opacity: float):
        """Handle surface opacity change."""
        if self.signals:
            self.signals.geologyOpacityChanged.emit(opacity)
        logger.debug(f"Surface opacity: {opacity}")

    def _on_solid_opacity_changed(self, opacity: float):
        """Handle solid opacity change."""
        if self.signals:
            self.signals.geologySolidsOpacityChanged.emit(opacity)
        logger.debug(f"Solid opacity: {opacity}")

    def _on_palette_changed(self, palette: str):
        """Handle color palette change."""
        if self.signals:
            self.signals.geologyColorPaletteChanged.emit(palette)
        logger.debug(f"Color palette: {palette}")

    def _on_reset_view(self):
        """Handle reset view button."""
        if self.signals:
            self.signals.geologyResetViewRequested.emit()
        logger.info("Geology reset view requested")

    def _on_clear(self):
        """Handle clear button."""
        if self.signals:
            self.signals.geologyClearRequested.emit()

        # Reset panel state
        self._geology_package = None
        self._surface_names = []
        self._solid_names = []
        self._surface_items.clear()
        self._solid_items.clear()
        self.surface_tree.clear()
        self.solid_tree.clear()
        self._apply_empty_state()
        logger.info("Geology cleared")

    # --- DATA LOADING ---

    def on_geology_package_loaded(self, package: Dict[str, Any]):
        """
        Handle geology package loaded from LoopStructural panel.

        Args:
            package: Dict containing 'surfaces', 'solids', 'report'
        """
        self._geology_package = package

        surfaces = package.get('surfaces', [])
        solids = package.get('solids', [])
        report = package.get('report') or package.get('audit_report')

        # Extract surface names
        self._surface_names = []
        for surface in surfaces:
            name = surface.get('name') or surface.get('formation') or f"Surface_{len(self._surface_names)}"
            self._surface_names.append(str(name))

        # Extract solid names
        self._solid_names = []
        for solid in solids:
            name = solid.get('unit_name') or solid.get('name') or f"Unit_{len(self._solid_names)}"
            self._solid_names.append(str(name))

        # Populate trees
        self._populate_surface_tree()
        self._populate_solid_tree()

        # Update contact stats
        if report and hasattr(report, 'misfit_data'):
            misfit_df = report.misfit_data
            self._contact_count = len(misfit_df) if misfit_df is not None else 0
            self._p90_error = getattr(report, 'p90_error', 0)
            self._mean_error = getattr(report, 'mean_residual', 0)
            self.contact_stats_label.setText(
                f"{self._contact_count} contacts | P90: {self._p90_error:.2f}m | Mean: {self._mean_error:.2f}m"
            )
        else:
            self._contact_count = 0
            self.contact_stats_label.setText("No contact data")

        # Update status
        total_layers = len(self._surface_names) + len(self._solid_names)
        self.status_badge.setText(f"{total_layers} Layers")
        self.status_badge.set_state(StatusBadge.State.SUCCESS)
        self.status_label.setText(
            f"{len(self._surface_names)} surfaces, {len(self._solid_names)} solids loaded"
        )
        self.surface_count_label.setText(f"{len(self._surface_names)} surfaces")
        self.solid_count_label.setText(f"{len(self._solid_names)} domains")

        # Enable controls
        self._apply_data_loaded_state()

        # CRITICAL FIX: Apply the current view mode after loading
        # The renderer loads ALL elements as visible, but the UI defaults to "Surfaces Only"
        # We need to sync the renderer visibility with the current view mode setting
        current_mode_text = self.view_mode_combo.currentText()
        if current_mode_text and self.signals:
            # Get the internal mode name
            mode_map = {
                "Surfaces Only": "surfaces_only",
                "Solids Only": "solids_only",
                "Contacts Only": "contacts_only",
                "Surfaces + Solids": "surfaces_solids",
                "All": "all"
            }
            mode = mode_map.get(current_mode_text, "surfaces_only")
            # Emit the view mode signal to update the renderer
            self.signals.geologyViewModeChanged.emit(mode)
            # Also update checkboxes to match
            self._update_checkboxes_for_view_mode(mode)
            logger.info(f"Applied initial view mode: {mode}")

        logger.info(
            f"Geology package loaded: {len(self._surface_names)} surfaces, "
            f"{len(self._solid_names)} solids, {self._contact_count} contacts"
        )

    def _populate_surface_tree(self):
        """Populate surface tree widget."""
        self.surface_tree.clear()
        self._surface_items.clear()

        self.surface_tree.blockSignals(True)
        for name in self._surface_names:
            item = QTreeWidgetItem([name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            self.surface_tree.addTopLevelItem(item)
            self._surface_items[name] = item
        self.surface_tree.blockSignals(False)

    def _populate_solid_tree(self):
        """Populate solid tree widget."""
        self.solid_tree.clear()
        self._solid_items.clear()

        self.solid_tree.blockSignals(True)
        for name in self._solid_names:
            item = QTreeWidgetItem([name])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            self.solid_tree.addTopLevelItem(item)
            self._solid_items[name] = item
        self.solid_tree.blockSignals(False)

    # --- STATE MANAGEMENT ---

    def _apply_empty_state(self):
        """Apply EMPTY state: Disable all controls."""
        self.view_mode_combo.setEnabled(False)
        self.all_surfaces_cb.setEnabled(False)
        self.all_solids_cb.setEnabled(False)
        self.show_contacts_cb.setEnabled(False)
        self.wireframe_cb.setEnabled(False)
        self.surface_opacity_slider.setEnabled(False)
        self.solid_opacity_slider.setEnabled(False)
        self.palette_combo.setEnabled(False)
        self.reset_view_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        self.status_badge.setText("No Data")
        self.status_badge.set_state(StatusBadge.State.NEUTRAL)
        self.status_label.setText("Build geological model to begin")
        self.surface_count_label.setText("0 surfaces")
        self.solid_count_label.setText("0 domains")
        self.contact_stats_label.setText("No contact data")

    def _apply_data_loaded_state(self):
        """Apply DATA_LOADED state: Enable all controls."""
        self.view_mode_combo.setEnabled(True)
        self.all_surfaces_cb.setEnabled(True)
        self.all_solids_cb.setEnabled(True)
        self.show_contacts_cb.setEnabled(True)
        self.wireframe_cb.setEnabled(True)
        self.surface_opacity_slider.setEnabled(True)
        self.solid_opacity_slider.setEnabled(True)
        self.palette_combo.setEnabled(True)
        self.reset_view_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    # --- PUBLIC API ---

    def set_renderer(self, renderer):
        """Set renderer reference."""
        self.renderer = renderer

    def get_view_mode(self) -> str:
        """Get current view mode."""
        mode_map = {
            "Surfaces Only": "surfaces_only",
            "Solids Only": "solids_only",
            "Contacts Only": "contacts_only",
            "Surfaces + Solids": "surfaces_solids",
            "All": "all"
        }
        return mode_map.get(self.view_mode_combo.currentText(), "surfaces_only")

    def get_surface_opacity(self) -> float:
        """Get surface opacity value."""
        return self.surface_opacity_slider.value()

    def get_solid_opacity(self) -> float:
        """Get solid opacity value."""
        return self.solid_opacity_slider.value()

    def is_contacts_visible(self) -> bool:
        """Check if contacts are visible."""
        return self.show_contacts_cb.isChecked()

    def is_wireframe_enabled(self) -> bool:
        """Check if wireframe is enabled."""
        return self.wireframe_cb.isChecked()

    def get_color_palette(self) -> str:
        """Get current color palette."""
        return self.palette_combo.currentText()

    def get_contact_stats(self) -> Dict[str, float]:
        """Get contact/misfit statistics."""
        return {
            'count': self._contact_count,
            'p90_error': self._p90_error,
            'mean_error': self._mean_error
        }
