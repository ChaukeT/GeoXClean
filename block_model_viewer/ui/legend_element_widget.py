"""
Individual legend element widget for the multi-element legend system.

Renders a single continuous gradient or discrete category list with
header controls for visibility toggle and removal.
"""

from __future__ import annotations

from typing import Optional, Dict, Union, Tuple, List, TYPE_CHECKING
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy
)
from PyQt6.QtGui import (
    QPainter, QLinearGradient, QColor, QPen, QFont, QFontMetrics,
    QBrush, QPainterPath
)
from PyQt6.QtCore import Qt, QRectF, QRect, QSize, pyqtSignal

from matplotlib import cm

from .legend_types import LegendElement, LegendElementType
from .legend_theme import get_legend_theme

if TYPE_CHECKING:
    pass


from .modern_styles import get_theme_colors, ModernColors
class LegendElementWidget(QFrame):
    """
    Individual legend element with header controls and content area.

    Supports both continuous (gradient) and discrete (category) rendering.
    """

    # Signals
    visibility_changed = pyqtSignal(str, bool)   # element_id, visible
    remove_requested = pyqtSignal(str)           # element_id
    category_toggled = pyqtSignal(str, object, bool)  # element_id, category, visible

    def __init__(self, element: LegendElement, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.element = element
        self._theme = get_legend_theme()
        self._hover_category = None
        self._cached_cmap = None

        self._setup_ui()



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
        """Build the widget layout with header and content."""
        self.setObjectName("legendElement")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header frame with controls
        self._header = QFrame()
        self._header.setObjectName("elementHeader")
        self._header.setFixedHeight(32)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        # Visibility toggle button (eye icon as text)
        self._visibility_btn = QPushButton()
        self._visibility_btn.setObjectName("visibilityBtn")
        self._visibility_btn.setFixedSize(24, 24)
        self._visibility_btn.setCheckable(True)
        self._visibility_btn.setChecked(self.element.visible)
        self._visibility_btn.clicked.connect(self._on_visibility_clicked)
        self._update_visibility_icon()
        header_layout.addWidget(self._visibility_btn)

        # Title label
        self._title_label = QLabel(self.element.title)
        self._title_label.setObjectName("titleLabel")
        header_layout.addWidget(self._title_label, 1)

        # Remove button
        self._remove_btn = QPushButton("x")
        self._remove_btn.setObjectName("removeBtn")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setToolTip("Remove from legend")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        header_layout.addWidget(self._remove_btn)

        layout.addWidget(self._header)

        # Content area for gradient/categories
        self._content = LegendContentWidget(self.element, self)
        self._content.category_clicked.connect(self._on_category_clicked)
        self._content.setStyleSheet("background-color: #1A1A1E;")
        layout.addWidget(self._content)

        # Apply styling
        self._apply_styling()

        # Set size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _apply_styling(self):
        """Apply dark theme styling directly to widgets."""
        # Style the main frame
        self.setStyleSheet("""
            LegendElementWidget {
                background-color: #1E1E22;
                border: 1px solid #4A4A4A;
                border-radius: 6px;
            }
        """)

        # Style header directly
        self._header.setStyleSheet("""
            QFrame {
                background-color: #333338;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                border-bottom: 1px solid #4A4A4A;
            }
        """)

        # Style title label
        self._title_label.setStyleSheet("""
            QLabel {
                color: #E8E8E8;
                font-weight: bold;
                font-size: 11px;
                background: transparent;
            }
        """)

        # Style buttons
        button_style = """
            QPushButton {
                background-color: #454550;
                border: 1px solid #666;
                border-radius: 4px;
                color: #DDD;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5A5A60;
                color: #FFF;
            }
        """
        self._visibility_btn.setStyleSheet(button_style + """
            QPushButton:checked {
                background-color: #3C7850;
                color: #8F8;
            }
        """)
        self._remove_btn.setStyleSheet(button_style + """
            QPushButton:hover {
                background-color: #964040;
                color: #FFF;
            }
        """)

    def _update_visibility_icon(self):
        """Update visibility button text based on state."""
        if self.element.visible:
            self._visibility_btn.setText("O")  # Open eye symbol
            self._visibility_btn.setToolTip("Click to hide")
        else:
            self._visibility_btn.setText("-")  # Closed eye symbol
            self._visibility_btn.setToolTip("Click to show")

    def _on_visibility_clicked(self):
        """Handle visibility toggle."""
        self.element.visible = self._visibility_btn.isChecked()
        self._update_visibility_icon()
        self._content.setVisible(self.element.visible)
        self.visibility_changed.emit(self.element.id, self.element.visible)
        self.updateGeometry()

    def _on_remove_clicked(self):
        """Handle remove button click."""
        self.remove_requested.emit(self.element.id)

    def _on_category_clicked(self, category):
        """Handle category click in discrete legend."""
        if category in self.element.category_visible:
            new_visible = not self.element.category_visible[category]
            self.element.category_visible[category] = new_visible
            self._content.update()
            self.category_toggled.emit(self.element.id, category, new_visible)

    def update_element(self, element: LegendElement):
        """Update the displayed element data."""
        self.element = element
        self._title_label.setText(element.title)
        self._visibility_btn.setChecked(element.visible)
        self._update_visibility_icon()
        self._content.element = element
        self._content.update()
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        """Provide size hint based on content."""
        header_height = 32
        content_height = self._content.sizeHint().height() if self.element.visible else 0
        return QSize(220, header_height + content_height + 2)

    def minimumSizeHint(self) -> QSize:
        """Minimum size."""
        return QSize(180, 34)


class LegendContentWidget(QWidget):
    """
    Content area widget that renders continuous or discrete legend.
    """

    category_clicked = pyqtSignal(object)  # category value

    def __init__(self, element: LegendElement, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.element = element
        self._theme = get_legend_theme()
        self._cached_cmap = None
        self._hover_category = None

        self.setMouseTracking(True)
        self._update_size()

    def _update_size(self):
        """Update fixed size based on content."""
        if self.element.element_type == LegendElementType.CONTINUOUS:
            self.setFixedHeight(100)
        else:
            # Height based on category count (max 8 visible at once)
            count = min(len(self.element.categories), 8)
            height = 8 + count * 22 + 8
            self.setFixedHeight(max(50, height))

    def paintEvent(self, event):
        """Render the legend content."""
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect()

        if self.element.element_type == LegendElementType.CONTINUOUS:
            self._draw_continuous(qp, rect)
        else:
            self._draw_discrete(qp, rect)

    def _draw_continuous(self, qp: QPainter, rect: QRect):
        """Render continuous scalar gradient."""
        if self.element.vmin is None or self.element.vmax is None:
            return
        if self.element.vmin >= self.element.vmax:
            return

        font = QFont(self._theme.font_family, 10)
        qp.setFont(font)
        fm = QFontMetrics(font)

        padding = 10
        bar_width = 20
        bar_left = padding
        bar_top = padding
        bar_height = rect.height() - 2 * padding
        bar_rect = QRectF(bar_left, bar_top, bar_width, bar_height)

        # Draw gradient bar
        self._draw_gradient_bar(qp, bar_rect)

        # Draw tick labels
        tick_count = min(self.element.tick_count, 5)
        qp.setPen(QPen(QColor(220, 220, 220)))

        for i in range(tick_count):
            n = i / (tick_count - 1) if tick_count > 1 else 0.5
            if self.element.reverse:
                n = 1.0 - n

            val = self.element.vmin + n * (self.element.vmax - self.element.vmin)
            y = bar_rect.top() + (1.0 - n) * bar_rect.height()

            # Format value
            if abs(val) >= 1000:
                label_text = f"{val:.0f}"
            elif abs(val) >= 10:
                label_text = f"{val:.1f}"
            elif abs(val) >= 1:
                label_text = f"{val:.2f}"
            else:
                label_text = f"{val:.2f}"

            label_x = bar_rect.right() + 8
            label_y = y + fm.ascent() / 3
            qp.drawText(int(label_x), int(label_y), label_text)

    def _draw_gradient_bar(self, qp: QPainter, rect: QRectF):
        """Render the gradient bar."""
        # Cache colormap
        cmap_name = self.element.cmap_name or 'viridis'
        if self._cached_cmap is None or getattr(self._cached_cmap, 'name', '') != cmap_name:
            try:
                self._cached_cmap = cm.get_cmap(cmap_name)
            except Exception:
                self._cached_cmap = cm.get_cmap('viridis')

        grad = QLinearGradient()
        grad.setStart(rect.bottomLeft())
        grad.setFinalStop(rect.topLeft())

        # Build gradient stops
        steps = 32
        for i in range(steps):
            n = i / (steps - 1)
            if self.element.reverse:
                n = 1.0 - n

            try:
                rgba = self._cached_cmap(n)
                if isinstance(rgba, (list, tuple, np.ndarray)):
                    rgba = np.asarray(rgba, dtype=float)
                    if rgba.size == 3:
                        rgba = np.append(rgba, 1.0)
                else:
                    rgba = np.array([1.0, 1.0, 1.0, 1.0])
            except Exception:
                rgba = np.array([1.0, 1.0, 1.0, 1.0])

            if len(rgba) >= 3 and max(rgba[:3]) > 1.01:
                rgba = rgba / 255.0

            color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
            grad.setColorAt(i / (steps - 1), color)

        qp.save()
        qp.setBrush(QBrush(grad))
        qp.setPen(QPen(QColor(60, 60, 60), 1))
        qp.drawRoundedRect(rect, 4, 4)
        qp.restore()

    def _draw_discrete(self, qp: QPainter, rect: QRect):
        """Render discrete category swatches."""
        if not self.element.categories:
            return

        font = QFont(self._theme.font_family, 10)
        qp.setFont(font)
        fm = QFontMetrics(font)

        padding = 8
        box_size = 14
        row_height = 20
        row_spacing = 2

        y = padding
        x = padding

        # Limit to 8 visible categories
        visible_categories = self.element.categories[:8]

        for category in visible_categories:
            visible = self.element.category_visible.get(category, True)

            # Color box
            box_rect = QRectF(x, y, box_size, box_size)
            color = self.element.category_colors.get(category, (0.5, 0.5, 0.5, 1.0))

            # Normalize color values
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                r, g, b = float(color[0]), float(color[1]), float(color[2])
                a = float(color[3]) if len(color) > 3 else 1.0
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                swatch_color = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            else:
                swatch_color = QColor(128, 128, 128, 255)

            # Dim if not visible
            if not visible:
                swatch_color.setAlpha(80)

            # Draw swatch
            qp.save()
            qp.setBrush(QBrush(swatch_color))
            qp.setPen(QPen(QColor(80, 80, 80), 1))
            qp.drawRoundedRect(box_rect, 3, 3)
            qp.restore()

            # Label
            label_text = str(self.element.category_labels.get(category, category))
            # Truncate long labels
            if len(label_text) > 20:
                label_text = label_text[:18] + ".."

            label_x = x + box_size + 6
            label_y = y + fm.ascent() - 1

            if visible:
                qp.setPen(QPen(QColor(220, 220, 220)))
            else:
                qp.setPen(QPen(QColor(100, 100, 100)))

            qp.drawText(int(label_x), int(label_y), label_text)

            # Hover highlight
            if self._hover_category == category:
                highlight_rect = QRectF(x - 2, y - 2, rect.width() - 2 * padding + 4, row_height)
                qp.fillRect(highlight_rect, QColor(255, 255, 255, 20))

            y += row_height + row_spacing

        # Show "+N more" if truncated
        if len(self.element.categories) > 8:
            more_count = len(self.element.categories) - 8
            qp.setPen(QPen(QColor(150, 150, 150)))
            qp.drawText(int(x), int(y + fm.ascent()), f"+{more_count} more...")

    def mousePressEvent(self, event):
        """Handle clicks on categories."""
        if self.element.element_type == LegendElementType.DISCRETE:
            category = self._get_category_at_pos(event.pos())
            if category is not None:
                self.category_clicked.emit(category)

    def mouseMoveEvent(self, event):
        """Track hover over categories."""
        if self.element.element_type == LegendElementType.DISCRETE:
            category = self._get_category_at_pos(event.pos())
            if category != self._hover_category:
                self._hover_category = category
                self.update()

    def leaveEvent(self, event):
        """Clear hover state on leave."""
        if self._hover_category is not None:
            self._hover_category = None
            self.update()

    def _get_category_at_pos(self, pos) -> Optional[object]:
        """Get category at mouse position."""
        if not self.element.categories:
            return None

        padding = 8
        row_height = 20
        row_spacing = 2
        y = padding

        visible_categories = self.element.categories[:8]
        for category in visible_categories:
            if y <= pos.y() < y + row_height:
                return category
            y += row_height + row_spacing

        return None

    def sizeHint(self) -> QSize:
        """Calculate size based on content."""
        if self.element.element_type == LegendElementType.CONTINUOUS:
            return QSize(180, 100)
        else:
            count = min(len(self.element.categories), 8)
            height = 8 + count * 22 + 8
            if len(self.element.categories) > 8:
                height += 20  # For "+N more" text
            return QSize(180, max(50, height))

    def minimumSizeHint(self) -> QSize:
        """Minimum size for the content."""
        return QSize(150, 50)
