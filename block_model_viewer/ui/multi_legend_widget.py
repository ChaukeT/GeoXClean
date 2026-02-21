"""
Multi-element legend container widget.

Provides a scrollable container with a toolbar for managing multiple
legend elements (both continuous and discrete) simultaneously.
"""

from __future__ import annotations

from typing import Optional, Dict, List, TYPE_CHECKING
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton,
    QFrame, QLabel, QSizePolicy, QToolButton, QMenu
)
from PyQt6.QtGui import QColor, QFont, QAction, QMouseEvent
from PyQt6.QtCore import Qt, QSize, pyqtSignal

from .legend_types import LegendElement, LegendElementType, MultiLegendConfig
from .legend_element_widget import LegendElementWidget
from .legend_theme import get_legend_theme

if TYPE_CHECKING:
    from ..visualization.renderer import Renderer

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class MultiLegendWidget(QFrame):
    """
    Container widget for multiple legend elements with add/remove capabilities.

    Layout:
    +------------------------------------------+
    |  LEGEND                    [+] [Settings]|  <- Toolbar
    +------------------------------------------+
    |  [LegendElementWidget 1]                 |  <- Scrollable content
    |  [LegendElementWidget 2]                 |
    |  ...                                     |
    +------------------------------------------+
    """

    # Signals
    element_added = pyqtSignal(str)           # element_id
    element_removed = pyqtSignal(str)         # element_id
    element_visibility_changed = pyqtSignal(str, bool)  # element_id, visible
    category_toggled = pyqtSignal(str, object, bool)  # element_id, category, visible
    config_changed = pyqtSignal(dict)         # Full config dict for persistence
    add_requested = pyqtSignal()              # Request to show add dialog

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = MultiLegendConfig()
        self._element_widgets: Dict[str, LegendElementWidget] = {}
        self._theme = get_legend_theme()

        # Dragging state
        self._dragging = False
        self._drag_start_pos = None

        self._setup_ui()
        self._apply_styling()



    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _setup_ui(self):
        """Build the widget layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Toolbar (draggable area)
        self._toolbar = QFrame()
        self._toolbar.setObjectName("toolbar")
        self._toolbar.setFixedHeight(36)
        self._toolbar.setCursor(Qt.CursorShape.SizeAllCursor)  # Indicate draggable
        toolbar_layout = QHBoxLayout(self._toolbar)
        toolbar_layout.setContentsMargins(12, 6, 12, 6)
        toolbar_layout.setSpacing(8)

        # Title
        title_label = QLabel("LEGEND")
        title_font = QFont(self._theme.font_family, self._theme.font_size)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #F5F5F5;")
        toolbar_layout.addWidget(title_label)

        toolbar_layout.addStretch()

        # Add button
        self._add_btn = QToolButton()
        self._add_btn.setText("+")
        self._add_btn.setToolTip("Add legend element")
        self._add_btn.setFixedSize(24, 24)
        self._add_btn.clicked.connect(self._on_add_clicked)
        toolbar_layout.addWidget(self._add_btn)

        # Settings button
        self._settings_btn = QToolButton()
        self._settings_btn.setText("\u2699")  # Gear symbol
        self._settings_btn.setToolTip("Legend settings")
        self._settings_btn.setFixedSize(24, 24)
        self._settings_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._create_settings_menu()
        toolbar_layout.addWidget(self._settings_btn)

        main_layout.addWidget(self._toolbar)

        # Scroll area for elements
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Container for element widgets
        self._elements_container = QWidget()
        self._elements_container.setObjectName("elementsContainer")
        self._elements_layout = QVBoxLayout(self._elements_container)
        self._elements_layout.setContentsMargins(8, 8, 8, 8)
        self._elements_layout.setSpacing(8)
        self._elements_layout.addStretch()

        self._scroll_area.setWidget(self._elements_container)
        main_layout.addWidget(self._scroll_area, 1)

        # Empty state label
        self._empty_label = QLabel("Click + to add legend elements")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888; padding: 20px;")
        self._elements_layout.insertWidget(0, self._empty_label)

        self.setMinimumSize(200, 150)
        self.resize(280, 300)

    def _apply_styling(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            MultiLegendWidget {
                background-color: rgba(15, 15, 20, 245);
                border: 1px solid #3C3C3C;
                border-radius: 12px;
            }
            QFrame#toolbar {
                background-color: rgba(35, 35, 40, 255);
                border-top-left-radius: 11px;
                border-top-right-radius: 11px;
                border-bottom: 1px solid #3C3C3C;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QToolButton {
                background-color: rgba(60, 60, 65, 200);
                border: 1px solid #555;
                border-radius: 4px;
                color: #DDD;
                font-size: 16px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: rgba(80, 80, 85, 255);
                border-color: #777;
            }
            QToolButton:pressed {
                background-color: rgba(50, 50, 55, 255);
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: rgba(40, 40, 45, 150);
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: rgba(100, 100, 105, 200);
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: rgba(130, 130, 135, 255);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def _create_settings_menu(self):
        """Create settings dropdown menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2D2D30;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                color: #DDD;
            }
            QMenu::item:selected {
                background-color: #3E3E42;
            }
        """)

        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self.clear_all)
        menu.addAction(clear_action)

        self._settings_btn.setMenu(menu)

    def _on_add_clicked(self):
        """Handle add button click."""
        self.add_requested.emit()

    def add_element(self, element: LegendElement) -> None:
        """
        Add or update a legend element.

        If an element with the same ID exists, it will be updated.
        """
        # Update config
        self.config.add_element(element)

        # Check if widget already exists
        if element.id in self._element_widgets:
            # Update existing widget
            self._element_widgets[element.id].update_element(element)
        else:
            # Create new widget
            widget = LegendElementWidget(element, self)
            widget.visibility_changed.connect(self._on_element_visibility_changed)
            widget.remove_requested.connect(self._on_element_remove_requested)
            widget.category_toggled.connect(self._on_category_toggled)

            self._element_widgets[element.id] = widget

            # Insert before the stretch
            count = self._elements_layout.count()
            self._elements_layout.insertWidget(count - 1, widget)

        # Hide empty label
        self._empty_label.hide()

        # Emit signals
        self.element_added.emit(element.id)
        self._emit_config_changed()

        logger.debug(f"Added legend element: {element.id} ({element.element_type.value})")

    def remove_element(self, element_id: str) -> bool:
        """
        Remove element by ID.

        Returns True if element was removed, False if not found.
        """
        if element_id not in self._element_widgets:
            return False

        # Remove widget
        widget = self._element_widgets.pop(element_id)
        self._elements_layout.removeWidget(widget)
        widget.deleteLater()

        # Update config
        self.config.remove_element(element_id)

        # Show empty label if no elements
        if not self._element_widgets:
            self._empty_label.show()

        # Emit signals
        self.element_removed.emit(element_id)
        self._emit_config_changed()

        logger.debug(f"Removed legend element: {element_id}")
        return True

    def clear_all(self) -> None:
        """Remove all legend elements."""
        for element_id in list(self._element_widgets.keys()):
            self.remove_element(element_id)

    def get_element(self, element_id: str) -> Optional[LegendElement]:
        """Get element by ID."""
        return self.config.get_element(element_id)

    def has_element(self, element_id: str) -> bool:
        """Check if element exists."""
        return element_id in self._element_widgets

    def get_element_ids(self) -> List[str]:
        """Get list of all element IDs."""
        return list(self._element_widgets.keys())

    def set_element_visibility(self, element_id: str, visible: bool) -> bool:
        """Set visibility for an element."""
        if element_id not in self._element_widgets:
            return False

        self.config.set_visibility(element_id, visible)
        widget = self._element_widgets[element_id]
        widget.element.visible = visible
        widget._visibility_btn.setChecked(visible)
        widget._update_visibility_icon()
        widget._content.setVisible(visible)

        self._emit_config_changed()
        return True

    def _on_element_visibility_changed(self, element_id: str, visible: bool):
        """Handle visibility change from element widget."""
        self.config.set_visibility(element_id, visible)
        self.element_visibility_changed.emit(element_id, visible)
        self._emit_config_changed()

    def _on_element_remove_requested(self, element_id: str):
        """Handle remove request from element widget."""
        self.remove_element(element_id)

    def _on_category_toggled(self, element_id: str, category, visible: bool):
        """Handle category toggle from element widget."""
        elem = self.config.get_element(element_id)
        if elem:
            elem.category_visible[category] = visible
            self.category_toggled.emit(element_id, category, visible)
            self._emit_config_changed()

    def _emit_config_changed(self):
        """Emit config change signal with current state."""
        self.config_changed.emit(self.config.to_dict())

    def load_config(self, config_dict: Dict) -> None:
        """Load configuration from dictionary."""
        self.clear_all()
        self.config = MultiLegendConfig.from_dict(config_dict)
        for element in self.config.elements:
            # Add widget without triggering add_element's config update
            widget = LegendElementWidget(element, self)
            widget.visibility_changed.connect(self._on_element_visibility_changed)
            widget.remove_requested.connect(self._on_element_remove_requested)
            widget.category_toggled.connect(self._on_category_toggled)

            self._element_widgets[element.id] = widget
            count = self._elements_layout.count()
            self._elements_layout.insertWidget(count - 1, widget)

        if self._element_widgets:
            self._empty_label.hide()

    def sizeHint(self) -> QSize:
        """Provide size hint."""
        return QSize(280, 350)

    def minimumSizeHint(self) -> QSize:
        """Provide minimum size."""
        return QSize(200, 150)

    def mousePressEvent(self, event):
        """Start dragging when clicking on the toolbar area."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Only drag if clicking on the toolbar area (top 40 pixels)
            if event.pos().y() <= 40:
                self._dragging = True
                self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Move the widget while dragging."""
        if self._dragging and self._drag_start_pos is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_start_pos
            self.move(new_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Stop dragging."""
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._drag_start_pos = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
