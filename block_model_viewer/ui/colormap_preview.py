"""
Colormap Preview — Visual colormap selector.

Shows a color gradient preview next to each colormap name in a QComboBox.
Drop-in replacement for plain text colormap combos.

Usage:
    from .colormap_preview import ColormapCombo

    self.colormap_combo = ColormapCombo()
    self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
    layout.addWidget(self.colormap_combo)
"""

from __future__ import annotations

from typing import Optional, List
import logging

import numpy as np

from PyQt6.QtCore import Qt, QRect, QSize
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QComboBox, QStyledItemDelegate, QStyleOptionViewItem, QWidget,
)

from ..visualization.color_mapper import safe_get_cmap

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# COLORMAP → RGB SAMPLING
# ═══════════════════════════════════════════════════════════════════

def _sample_colormap(name: str, n: int = 64) -> List[QColor]:
    """
    Sample a matplotlib colormap and return QColor list.

    Falls back to grayscale if matplotlib is not available.
    """
    try:
        cmap = safe_get_cmap(name)
        colors = []
        for i in range(n):
            t = i / max(1, n - 1)
            r, g, b, a = cmap(t)
            colors.append(QColor(int(r * 255), int(g * 255), int(b * 255)))
        return colors
    except Exception:
        # Fallback: grayscale gradient
        return [QColor(int(i * 255 / max(1, n - 1)),
                       int(i * 255 / max(1, n - 1)),
                       int(i * 255 / max(1, n - 1)))
                for i in range(n)]


def _make_gradient_pixmap(colors: List[QColor], width: int = 80, height: int = 14) -> QPixmap:
    """Create a horizontal gradient pixmap from a list of QColors."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    if not colors:
        painter.end()
        return pixmap

    # Draw color strips
    step = max(1, width / len(colors))
    for i, color in enumerate(colors):
        x = int(i * step)
        w = int(step) + 1  # +1 to avoid gaps
        painter.fillRect(x, 0, w, height, color)

    # Rounded corners mask
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    mask = QPixmap(width, height)
    mask.fill(Qt.GlobalColor.transparent)
    mask_painter = QPainter(mask)
    mask_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    mask_painter.setBrush(Qt.GlobalColor.white)
    mask_painter.setPen(Qt.PenStyle.NoPen)
    mask_painter.drawRoundedRect(0, 0, width, height, 3, 3)
    mask_painter.end()
    painter.drawPixmap(0, 0, mask)

    painter.end()
    return pixmap


# ═══════════════════════════════════════════════════════════════════
# CACHE — avoid resampling on every paint
# ═══════════════════════════════════════════════════════════════════

_pixmap_cache: dict[str, QPixmap] = {}


def get_colormap_pixmap(name: str, width: int = 80, height: int = 14) -> QPixmap:
    """Get cached colormap gradient pixmap."""
    key = f"{name}_{width}_{height}"
    if key not in _pixmap_cache:
        colors = _sample_colormap(name, n=min(width, 64))
        _pixmap_cache[key] = _make_gradient_pixmap(colors, width, height)
    return _pixmap_cache[key]


# ═══════════════════════════════════════════════════════════════════
# DELEGATE — paints gradient + text in combo dropdown
# ═══════════════════════════════════════════════════════════════════

class ColormapDelegate(QStyledItemDelegate):
    """
    Custom delegate that draws [gradient] [name] for each colormap.
    """

    GRADIENT_WIDTH = 80
    GRADIENT_HEIGHT = 14
    SPACING = 10

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index):
        # Draw default background (selection highlight, hover, etc.)
        self.initStyleOption(option, index)
        super().paint(painter, option, index)

        text = index.data(Qt.ItemDataRole.DisplayRole)
        if not text or text.startswith("---"):
            # Separator item — let default painting handle it
            return

        painter.save()

        rect = option.rect
        y_center = rect.center().y()

        # Draw gradient
        grad_rect = QRect(
            rect.left() + 6,
            y_center - self.GRADIENT_HEIGHT // 2,
            self.GRADIENT_WIDTH,
            self.GRADIENT_HEIGHT,
        )
        pixmap = get_colormap_pixmap(text, self.GRADIENT_WIDTH, self.GRADIENT_HEIGHT)
        painter.drawPixmap(grad_rect, pixmap)

        # Draw text after gradient
        text_rect = QRect(
            grad_rect.right() + self.SPACING,
            rect.top(),
            rect.width() - self.GRADIENT_WIDTH - self.SPACING - 12,
            rect.height(),
        )
        painter.setPen(option.palette.text().color())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        size = super().sizeHint(option, index)
        return QSize(
            max(size.width(), self.GRADIENT_WIDTH + self.SPACING + 80),
            max(size.height(), self.GRADIENT_HEIGHT + 8),
        )


# ═══════════════════════════════════════════════════════════════════
# COMBO — ready-to-use colormap selector
# ═══════════════════════════════════════════════════════════════════

class ColormapCombo(QComboBox):
    """
    QComboBox with visual colormap gradient previews.

    Usage:
        combo = ColormapCombo()
        # or with custom maps:
        combo = ColormapCombo(
            categorical=["tab10", "tab20", "Set1"],
            continuous=["turbo", "viridis", "plasma"],
        )
    """

    # Default colormap lists
    DEFAULT_CATEGORICAL = ["tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "Accent", "Dark2"]
    DEFAULT_CONTINUOUS = [
        "turbo", "viridis", "plasma", "inferno", "magma", "cividis",
        "coolwarm", "RdYlBu", "Spectral", "RdBu", "PiYG",
        "hot", "copper", "bone", "terrain", "gist_earth",
        "jet", "rainbow", "gnuplot", "gnuplot2",
    ]

    def __init__(
        self,
        categorical: Optional[List[str]] = None,
        continuous: Optional[List[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        # Apply delegate
        self._delegate = ColormapDelegate(self)
        self.setItemDelegate(self._delegate)
        self.setMinimumHeight(28)
        self.setObjectName("panelCombo")

        # Populate
        cat_maps = categorical or self.DEFAULT_CATEGORICAL
        cont_maps = continuous or self.DEFAULT_CONTINUOUS

        self._add_separator("Categorical")
        self.addItems(cat_maps)
        self._add_separator("Continuous")
        self.addItems(cont_maps)

        # Default selection
        if "turbo" in cont_maps:
            self.setCurrentText("turbo")

    def _add_separator(self, label: str) -> None:
        """Add a non-selectable separator/header item."""
        self.addItem(f"--- {label} ---")
        idx = self.count() - 1
        item = self.model().item(idx)
        if item:
            item.setEnabled(False)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

    def set_colormaps(
        self,
        categorical: List[str],
        continuous: List[str],
    ) -> None:
        """Replace all colormaps."""
        current = self.currentText()
        self.clear()
        self._add_separator("Categorical")
        self.addItems(categorical)
        self._add_separator("Continuous")
        self.addItems(continuous)
        # Restore selection if possible
        idx = self.findText(current)
        if idx >= 0:
            self.setCurrentIndex(idx)
