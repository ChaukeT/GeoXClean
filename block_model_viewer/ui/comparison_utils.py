"""
Comparison utilities for multi-source block model comparison.

Provides:
- ComparisonColors: Color palette for distinguishing sources
- SourceSelectionWidget: Reusable checkbox list for selecting multiple sources
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QFrame,
    QLabel, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


class ComparisonColors:
    """Color palette for multi-source comparison plots.

    Uses a color-blind friendly palette that works well on dark backgrounds.
    """

    # Primary comparison colors (color-blind friendly)
    PALETTE = [
        "#0e7aca",  # Blue (primary accent)
        "#f0883e",  # Orange
        "#26a69a",  # Teal
        "#f44336",  # Red
        "#9c27b0",  # Purple
        "#4caf50",  # Green
        "#ff9800",  # Amber
        "#795548",  # Brown
    ]

    # Line styles for additional differentiation
    LINE_STYLES = [
        "-",        # Solid
        "--",       # Dashed
        "-.",       # Dash-dot
        ":",        # Dotted
    ]

    # Markers for scatter/points
    MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p"]

    # Alpha values for fills
    FILL_ALPHA = 0.2
    LINE_ALPHA = 0.9

    @classmethod
    def get_color(cls, index: int) -> str:
        """Get color for source at given index."""
        return cls.PALETTE[index % len(cls.PALETTE)]

    @classmethod
    def get_line_style(cls, index: int) -> str:
        """Get line style for source at given index."""
        return cls.LINE_STYLES[index % len(cls.LINE_STYLES)]

    @classmethod
    def get_marker(cls, index: int) -> str:
        """Get marker for source at given index."""
        return cls.MARKERS[index % len(cls.MARKERS)]

    @classmethod
    def get_style(cls, index: int) -> Dict[str, Any]:
        """Get complete style dict for source at given index.

        Returns dict with: color, linestyle, marker, alpha
        """
        return {
            "color": cls.get_color(index),
            "linestyle": cls.get_line_style(index),
            "marker": cls.get_marker(index),
            "alpha": cls.LINE_ALPHA,
            "fill_alpha": cls.FILL_ALPHA,
        }

    @classmethod
    def get_fill_color(cls, index: int, alpha: float = None) -> Tuple[float, float, float, float]:
        """Get RGBA tuple for filled regions.

        Args:
            index: Source index
            alpha: Override alpha value (default: FILL_ALPHA)

        Returns:
            RGBA tuple (0-1 range)
        """
        import matplotlib.colors as mcolors
        hex_color = cls.get_color(index)
        rgb = mcolors.to_rgb(hex_color)
        return (*rgb, alpha if alpha is not None else cls.FILL_ALPHA)


class SourceCheckBox(QCheckBox):
    """Custom checkbox for source selection with metadata."""

    def __init__(self, source_key: str, display_name: str, block_count: int = 0, parent=None):
        super().__init__(parent)
        self.source_key = source_key
        self.display_name = display_name
        self.block_count = block_count

        # Format display text
        if block_count > 0:
            text = f"{display_name} ({block_count:,} blocks)"
        else:
            text = display_name
        self.setText(text)

        # Style
        self.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                padding: 4px 8px;
                spacing: 8px;
            }
            QCheckBox:hover {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #555;
                border-radius: 3px;
                background: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background: #0e7aca;
                border-color: #0e7aca;
            }
            QCheckBox::indicator:checked:hover {
                background: #1e8ada;
                border-color: #1e8ada;
            }
        """)


class SourceSelectionWidget(QWidget):
    """Reusable widget for multi-source selection with checkbox list.

    Provides:
    - Comparison mode toggle
    - Scrollable checkbox list of available sources
    - Selection count display
    - Signal when selection changes

    Usage:
        widget = SourceSelectionWidget()
        widget.update_sources({
            'sgsim_mean_FE': {'display_name': 'SGSIM Mean (FE)', 'block_count': 125000},
            'kriging_FE': {'display_name': 'Ordinary Kriging', 'block_count': 125000},
        })
        widget.sources_changed.connect(self._on_sources_changed)
    """

    # Emitted when selection changes, with list of selected source keys
    sources_changed = pyqtSignal(list)

    # Emitted when comparison mode is toggled
    comparison_mode_changed = pyqtSignal(bool)

    # Maximum recommended sources for comparison
    MAX_SOURCES = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._checkboxes: Dict[str, SourceCheckBox] = {}
        self._comparison_mode = False
        self._setup_ui()

        # Ensure the widget can expand when needed
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Comparison mode toggle
        self.compare_mode_check = QCheckBox("Enable Comparison Mode")
        self.compare_mode_check.setStyleSheet("""
            QCheckBox {
                color: #0e7aca;
                font-weight: bold;
                padding: 4px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        self.compare_mode_check.toggled.connect(self._on_compare_mode_toggled)
        layout.addWidget(self.compare_mode_check)

        # Source list frame (hidden by default)
        self.source_list_frame = QFrame()
        self.source_list_frame.setMinimumHeight(80)
        self.source_list_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        self.source_list_frame.setVisible(False)

        source_frame_layout = QVBoxLayout(self.source_list_frame)
        source_frame_layout.setContentsMargins(4, 4, 4, 4)
        source_frame_layout.setSpacing(2)

        # Scroll area for checkboxes
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setMinimumHeight(50)
        self.scroll_area.setMaximumHeight(200)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
        """)

        # Container for checkboxes
        self.checkbox_container = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_container)
        self.checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.checkbox_layout.setSpacing(2)
        self.checkbox_layout.addStretch()

        self.scroll_area.setWidget(self.checkbox_container)
        source_frame_layout.addWidget(self.scroll_area)

        # Selection count label
        self.selection_label = QLabel("Selected: 0 sources")
        self.selection_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        source_frame_layout.addWidget(self.selection_label)

        layout.addWidget(self.source_list_frame)

    def _on_compare_mode_toggled(self, checked: bool):
        """Handle comparison mode toggle."""
        self._comparison_mode = checked
        self.source_list_frame.setVisible(checked)
        logger.info(f"SourceSelectionWidget: Comparison mode toggled to {checked}, {len(self._checkboxes)} checkboxes available")
        self.comparison_mode_changed.emit(checked)

        if not checked:
            # Clear all selections when disabling comparison mode
            for checkbox in self._checkboxes.values():
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
            self._update_selection_label()
            self.sources_changed.emit([])

    def _on_checkbox_toggled(self, checked: bool):
        """Handle checkbox state change."""
        selected = self.get_selected_sources()

        # Warn if too many selected
        if len(selected) > self.MAX_SOURCES:
            sender = self.sender()
            if sender and isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
                logger.warning(f"Maximum {self.MAX_SOURCES} sources can be compared")
                return

        self._update_selection_label()
        self.sources_changed.emit(selected)

    def _update_selection_label(self):
        """Update the selection count label."""
        count = len(self.get_selected_sources())
        if count == 0:
            self.selection_label.setText("Selected: 0 sources")
            self.selection_label.setStyleSheet("color: #888; font-size: 11px; padding: 4px;")
        elif count == 1:
            self.selection_label.setText("Selected: 1 source (need 2+ for comparison)")
            self.selection_label.setStyleSheet("color: #f0883e; font-size: 11px; padding: 4px;")
        else:
            self.selection_label.setText(f"Selected: {count} sources")
            self.selection_label.setStyleSheet("color: #4caf50; font-size: 11px; padding: 4px;")

    def update_sources(self, sources: Dict[str, Dict[str, Any]]):
        """Update the available sources.

        Args:
            sources: Dict mapping source_key to source info dict.
                     Each source info should have:
                     - 'display_name': str - Human-readable name
                     - 'block_count': int - Number of blocks (optional)
                     - 'df': DataFrame - The data (optional, for reference)
        """
        # Store sources
        self._sources = sources

        # Clear existing checkboxes
        for checkbox in self._checkboxes.values():
            checkbox.deleteLater()
        self._checkboxes.clear()

        # Remove stretch from layout
        while self.checkbox_layout.count() > 0:
            item = self.checkbox_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add checkboxes for each source
        for source_key, info in sources.items():
            display_name = info.get('display_name', source_key)
            block_count = info.get('block_count', 0)

            checkbox = SourceCheckBox(source_key, display_name, block_count)
            checkbox.toggled.connect(self._on_checkbox_toggled)

            self._checkboxes[source_key] = checkbox
            self.checkbox_layout.addWidget(checkbox)

        # Add stretch at end
        self.checkbox_layout.addStretch()

        self._update_selection_label()
        logger.info(f"SourceSelectionWidget: Updated with {len(sources)} sources: {list(sources.keys())}")

    def get_selected_sources(self) -> List[str]:
        """Return list of selected source keys."""
        return [
            key for key, checkbox in self._checkboxes.items()
            if checkbox.isChecked()
        ]

    def get_selected_source_info(self) -> List[Dict[str, Any]]:
        """Return list of info dicts for selected sources."""
        selected_keys = self.get_selected_sources()
        return [
            {**self._sources[key], 'key': key}
            for key in selected_keys
            if key in self._sources
        ]

    def is_comparison_mode(self) -> bool:
        """Return True if comparison mode is enabled."""
        return self._comparison_mode

    def set_comparison_mode(self, enabled: bool):
        """Set comparison mode state."""
        self.compare_mode_check.setChecked(enabled)

    def select_source(self, source_key: str, selected: bool = True):
        """Programmatically select/deselect a source."""
        if source_key in self._checkboxes:
            self._checkboxes[source_key].setChecked(selected)

    def clear_selection(self):
        """Clear all selections."""
        for checkbox in self._checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        self._update_selection_label()
        self.sources_changed.emit([])


def create_comparison_legend(ax, sources: List[str], colors: List[str] = None):
    """Create a styled legend for comparison plot.

    Args:
        ax: Matplotlib axes
        sources: List of source names
        colors: Optional list of colors (uses ComparisonColors if not provided)

    Returns:
        Legend object
    """
    from matplotlib.lines import Line2D

    if colors is None:
        colors = [ComparisonColors.get_color(i) for i in range(len(sources))]

    lines = []
    for i, (source, color) in enumerate(zip(sources, colors)):
        line = Line2D(
            [0], [0],
            color=color,
            linestyle=ComparisonColors.get_line_style(i),
            linewidth=2,
            marker=ComparisonColors.get_marker(i),
            markersize=6,
            label=source
        )
        lines.append(line)

    legend = ax.legend(
        handles=lines,
        loc='upper right',
        frameon=True,
        facecolor='#2d2d2d',
        edgecolor='#444',
        fontsize=9,
        labelcolor='#e0e0e0'
    )
    legend.get_frame().set_alpha(0.9)
    return legend


def format_comparison_table(data: Dict[str, Dict[str, float]],
                           stats_order: List[str] = None) -> str:
    """Format comparison statistics as a markdown-style table.

    Args:
        data: Dict mapping source_name to dict of statistics
        stats_order: Optional list of statistic names in display order

    Returns:
        Formatted table string
    """
    if not data:
        return "No data for comparison"

    sources = list(data.keys())

    # Get all stats from first source if order not provided
    if stats_order is None:
        stats_order = list(next(iter(data.values())).keys())

    # Build header
    header = "| Statistic |"
    for source in sources:
        header += f" {source[:20]} |"

    # Build separator
    sep = "|" + "-" * 12 + "|"
    for _ in sources:
        sep += "-" * 15 + "|"

    # Build rows
    rows = []
    for stat in stats_order:
        row = f"| {stat:<10} |"
        for source in sources:
            value = data[source].get(stat, float('nan'))
            if isinstance(value, float):
                row += f" {value:>13.4f} |"
            else:
                row += f" {str(value):>13} |"
        rows.append(row)

    return "\n".join([header, sep] + rows)
