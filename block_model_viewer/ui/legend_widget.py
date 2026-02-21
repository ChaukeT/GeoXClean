"""
High-performance, high-DPI PyQt6 legend widget for 3D scalar and categorical data.
Manual QPainter-based rendering with full control over appearance and interactivity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union, Callable, TYPE_CHECKING, Any
import numpy as np
from enum import Enum
import weakref

from PyQt6.QtWidgets import (
    QWidget, QMenu, QColorDialog, QMessageBox, QFileDialog, QToolTip, QInputDialog
)
from PyQt6.QtGui import (
    QPainter, QLinearGradient, QColor, QPen, QFont, QFontMetrics, QMouseEvent,
    QAction, QPixmap, QPainterPath, QBrush
)
from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, QSize, QRect, pyqtSignal

from matplotlib import cm
import matplotlib.colors as mcolors

from .legend_logging import get_legend_logger
from .legend_theme import get_legend_theme, LEGEND_THEME
from .legend_types import LegendElement, LegendElementType
from ..controllers.app_state import AppState

if TYPE_CHECKING:
    from ..visualization.legend_manager import LegendManager  # pragma: no cover

from .modern_styles import get_theme_colors, ModernColors
logger = get_legend_logger("widget")


# ============================================================================
# Data Models
# ============================================================================

class LegendType(Enum):
    """Legend rendering mode."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass
class LegendConfig:
    """Configuration for legend rendering."""
    type: LegendType = LegendType.CONTINUOUS
    title: str = ""
    
    # Continuous mode parameters
    vmin: float = 0.0
    vmax: float = 1.0
    cmap_name: str = "viridis"
    tick_count: int = 6
    log_scale: bool = False
    reverse: bool = False
    
    # Discrete mode parameters
    categories: List[Union[str, int, float]] = field(default_factory=list)
    category_colors: Dict[Union[str, int, float], Tuple[float, float, float, float]] = field(default_factory=dict)
    category_labels: Dict[Union[str, int, float], str] = field(default_factory=dict)  # Display labels (code -> label alias)
    
    # Styling
    font_size: int = 13
    font_family: str = "Segoe UI"  # Modern default on Windows; fallback handled by Qt
    modern: bool = True
    background_color: QColor = field(default_factory=lambda: QColor(15, 15, 20, 255))
    text_color: QColor = field(default_factory=lambda: QColor(245, 245, 245))
    border_color: QColor = field(default_factory=lambda: QColor(60, 60, 60))
    enable_shadow: bool = True
    # Outline toggle (draw border around legend)
    draw_outline: bool = True
    # Card/bar styling
    card_corner_radius: float = 20.0
    bar_corner_radius: float = 10.0
    bar_outline_color: QColor = field(default_factory=lambda: QColor(20, 20, 20))
    bar_outline_width: float = 2.0
    swatch_corner_radius: float = 6.0
    tick_color: QColor = field(default_factory=lambda: QColor(235, 235, 235))
    tick_width: float = 1.8
    title_bold: bool = True
    subtitle: str = ""
    # Label formatting
    label_decimals: Optional[int] = None
    label_thousands_sep: bool = False
    # Brightness control for rendered colors (100 = original)
    colormap_brightness: int = 160
    
    # Markers (for continuous)
    markers: List[float] = field(default_factory=list)  # Additional value markers
    show_mean: bool = False
    show_median: bool = False
    
    def get_marker_values(self, data: Optional[np.ndarray] = None) -> List[float]:
        """Get all marker values including computed statistics."""
        markers = list(self.markers)
        if data is not None and len(data) > 0:
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                if self.show_mean:
                    markers.append(float(np.mean(finite_data)))
                if self.show_median:
                    markers.append(float(np.median(finite_data)))
        return sorted(set(markers))


# ============================================================================
# Core Legend Widget
# ============================================================================

class LegendWidget(QWidget):
    def set_layer_properties(self, properties: list, toggled: dict = None):
        """Set the list of properties for the current layer and their toggle states."""
        self._layer_properties = properties
        self._property_toggled = toggled or {p: True for p in properties}
        logger.debug(
            "Layer properties updated (count=%s, toggled_overrides=%s)",
            len(properties),
            bool(toggled),
        )
        self.update()

    def toggle_property(self, property_name: str):
        """Toggle visibility for a property in the legend."""
        if hasattr(self, '_property_toggled') and property_name in self._property_toggled:
            self._property_toggled[property_name] = not self._property_toggled[property_name]
            logger.debug(
                "Property toggle changed (%s -> %s)",
                property_name,
                self._property_toggled[property_name],
            )
            self.update()
            # Emit a signal if needed (can be added for integration)

    """
    High-performance, interactive legend widget for PyQt6/PyVista applications.
    
    Features:
    - Continuous scalar gradients with customizable colormaps
    - Discrete categorical legends with per-category colors
    - High-DPI support with pixel-perfect rendering
    - Interactive hover tooltips and value readout
    - Click-to-isolate value ranges (continuous mode)
    - Click-to-toggle category visibility (discrete mode)
    - Export to PNG at any DPI
    - Dynamic resizing and theming
    """
    
    # Signals
    value_range_changed = pyqtSignal(float, float)  # vmin, vmax
    colormap_changed = pyqtSignal(str)              # new colormap name
    reverse_toggled = pyqtSignal(bool)              # reverse on/off
    orientation_changed = pyqtSignal(str)           # 'vertical' or 'horizontal'
    category_toggled = pyqtSignal(object, bool)  # category, visible
    category_label_changed = pyqtSignal(object, str)  # category code, new label
    category_color_changed = pyqtSignal(object, tuple)  # category, new RGBA color tuple
    legend_clicked = pyqtSignal()
    background_color_changed = pyqtSignal(QColor)
    dock_requested = pyqtSignal(str)
    floating_position_changed = pyqtSignal(int, int)
    size_changed = pyqtSignal(int, int)

    # Multi-element signals
    element_added = pyqtSignal(str)           # element_id
    element_removed = pyqtSignal(str)         # element_id
    element_visibility_changed = pyqtSignal(str, bool)  # element_id, visible
    add_element_requested = pyqtSignal()      # request to show add dialog
    mode_changed = pyqtSignal(bool)           # True = multi-mode, False = classic mode
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration
        self.config = LegendConfig()
        self._theme = get_legend_theme()
        logger.debug("LegendWidget initialised (parent=%s)", parent.__class__.__name__ if parent else None)
        
        # Ensure widget is deleted on close to avoid lingering after app exits
        try:
            from PyQt6.QtCore import Qt
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        except Exception:
            pass
        
        # Apply theme defaults
        self.config.font_family = self._theme.font_family
        self.config.font_size = self._theme.font_size
        self.config.background_color = self._theme.background_color
        self.config.text_color = self._theme.text_color
        self.config.border_color = self._theme.border_color
        self.config.bar_outline_color = self._theme.bar_outline_color
        self.config.bar_corner_radius = self._theme.bar_corner_radius
        self.config.swatch_corner_radius = self._theme.swatch_corner_radius
        self.config.card_corner_radius = self._theme.card_corner_radius
        self.config.enable_shadow = self._theme.enable_shadow
        
        # Rendering state
        self._orientation = "vertical"  # "vertical" or "horizontal"
        # Use theme defaults
        self._padding = self._theme.padding
        self._bar_width = self._theme.gradient_bar_width
        self._tick_length = 6
        self._tick_spacing = 4  # Space between tick and label
        self._category_box_size = self._theme.swatch_size
        self._dock_anchor = "floating"
        self._snap_threshold = 60  # pixels
        
        # Interaction state
        self._hover_value = None
        self._hover_category = None
        self._hover_position = None
        self._dragging = False
        self._drag_start_pos = None
        self._resizing = False
        self._resize_start_size = None
        self._resize_start_pos = None
        self._resize_handle_size = 12
        
        # Visibility state (discrete mode)
        self._category_visible = {}
        
        # Cached rendering data
        self._cached_data = None
        self._cached_cmap = None
        self._custom_colormap = None
        self._custom_color_samples = None
        # Empty-state message (when no data to show)
        self._empty_message = None

        # Multi-element mode state
        self._multi_mode = False                    # Whether multi-element mode is active
        self._elements: List[LegendElement] = []   # List of legend elements
        self._toolbar_height = 36                   # Height of the toolbar with [+] button
        self._element_header_height = 28            # Height of each element's header row
        self._hover_element_id: Optional[str] = None  # Element being hovered
        self._hover_button: Optional[str] = None    # Button being hovered ('add', 'remove', 'visibility')
        self._element_rects: Dict[str, QRect] = {}  # Bounding rects for elements
        self._button_rects: Dict[str, QRect] = {}   # Bounding rects for buttons
        
        # Setup
        self.setMouseTracking(True)
        self.setMinimumSize(200, 120)
        self.resize(300, 200)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Use Tool window flag to ensure it stays on top and doesn't interfere with main window
        # Tool windows stay on top of their parent but don't appear in taskbar
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.Tool | 
            Qt.WindowType.WindowStaysOnTopHint
        )
        # Set explicit context menu policy
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        
        # Tooltip
        self.setToolTip("Legend: Hover for values, click to interact")
        self._legend_manager: Optional["LegendManager"] = None
        self._legend_payload: Optional[Dict[str, Any]] = None
        
        # Application state tracking - legend hidden in EMPTY state
        self._app_state: AppState = AppState.EMPTY
        self._should_be_visible: bool = False  # Track intended visibility separately

    # --------------------------------------------------------------------
    # Docking helpers
    # --------------------------------------------------------------------


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
    def set_current_anchor(self, anchor: str) -> None:
        """Update the active anchor identifier (for menu highlighting)."""
        self._dock_anchor = anchor or "floating"

    # --------------------------------------------------------------------
    # Manager bindings
    # --------------------------------------------------------------------
    def bind_manager(self, manager: Optional["LegendManager"]) -> None:
        """Attach/detach the LegendManager that owns this widget."""
        if getattr(self, "_legend_manager", None) is manager:
            return
        self._detach_manager()
        self._legend_manager = manager
        if manager is None:
            return
        try:
            manager.legend_changed.connect(self.on_legend_changed)
        except Exception:
            pass
        try:
            manager.visibility_changed.connect(self._on_manager_visibility_changed)
        except Exception:
            pass
        try:
            state = manager.get_state()
            if state:
                self.on_legend_changed(state)
        except Exception:
            pass

    def _detach_manager(self) -> None:
        manager = getattr(self, "_legend_manager", None)
        if manager is None:
            return
        try:
            manager.legend_changed.disconnect(self.on_legend_changed)
        except Exception:
            pass
        try:
            manager.visibility_changed.disconnect(self._on_manager_visibility_changed)
        except Exception:
            pass
        self._legend_manager = None

    def _on_manager_visibility_changed(self, visible: bool) -> None:
        # Track intended visibility - actual visibility depends on app state
        self._should_be_visible = bool(visible)
        self._update_visibility_for_state()
    
    # =========================================================================
    # Application State Handling
    # =========================================================================
    
    def on_app_state_changed(self, state: int) -> None:
        """
        Handle application state changes.
        
        Legend is hidden in EMPTY state, visible in RENDERED state.
        
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
        logger.debug(f"LegendWidget: State changed {old_state.name} -> {new_state.name}")
        
        self._update_visibility_for_state()
    
    def _update_visibility_for_state(self) -> None:
        """Update legend visibility based on app state and intended visibility."""
        # Legend should only be visible if:
        # 1. App state allows it (RENDERED state)
        # 2. Manager says it should be visible
        should_show = (
            self._app_state == AppState.RENDERED and 
            self._should_be_visible
        )
        self.setVisible(should_show)

    def on_legend_changed(self, payload: Dict[str, Any]) -> None:
        """Update the widget configuration when metadata changes."""
        self.refresh_display(payload)
    
    def refresh_display(self, payload: Dict[str, Any]) -> None:
        """
        Unified refresh method - updates widget from LegendManager payload.
        
        This is the single entry point for updating the legend display.
        All legend updates should go through this method.
        
        Args:
            payload: Dictionary containing legend metadata from LegendManager
        """
        if not payload:
            return
        
        self._legend_payload = payload
        title = payload.get("title") or payload.get("property") or ""
        if title:
            self.config.title = title
        
        vmin = payload.get("vmin")
        vmax = payload.get("vmax")
        if vmin is not None:
            try:
                self.config.vmin = float(vmin)
            except Exception:
                pass
        if vmax is not None:
            try:
                self.config.vmax = float(vmax)
            except Exception:
                pass
        
        cmap = payload.get("colormap")
        if cmap:
            self.config.cmap_name = cmap
        
        categories = payload.get("categories") or []
        if categories:
            self.config.type = LegendType.DISCRETE
            self.config.categories = list(categories)
            # Also set category colors if provided in payload
            category_colors = payload.get("category_colors")
            if category_colors:
                # Normalize colors to RGBA float tuples
                normalized_colors = {}
                for cat, color in category_colors.items():
                    if isinstance(color, (tuple, list)):
                        arr = list(color)
                        # Ensure 4 elements (RGBA)
                        if len(arr) == 3:
                            arr.append(1.0)
                        # Ensure values are in 0-1 range
                        if any(v > 1.0 for v in arr[:3]):
                            arr = [v / 255.0 for v in arr[:3]] + [arr[3] if len(arr) > 3 else 1.0]
                        normalized_colors[cat] = tuple(float(v) for v in arr[:4])
                    elif isinstance(color, str):
                        qc = QColor(color)
                        if qc.isValid():
                            normalized_colors[cat] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                        else:
                            normalized_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                    else:
                        normalized_colors[cat] = (0.5, 0.5, 0.5, 1.0)
                self.config.category_colors = normalized_colors
                logger.debug(f"refresh_display: Set {len(normalized_colors)} category colors")
        else:
            self.config.type = LegendType.CONTINUOUS
            self.config.categories = []
        
        # Ensure fully opaque background (Step 9 requirement)
        bg_color = self.config.background_color
        if isinstance(bg_color, QColor):
            bg_color.setAlpha(255)  # Fully opaque
            self.config.background_color = bg_color
        
        self.update()

    def apply_layout(
        self,
        *,
        anchor: str = "floating",
        position: Optional[Tuple[int, int]] = None,
        margin: int = 24,
        size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Apply layout instructions originating from the legend manager."""
        if size:
            self.resize(max(size[0], self.minimumWidth()), max(size[1], self.minimumHeight()))

        parent = self.parentWidget()
        if parent is None:
            if position:
                self.move(position[0], position[1])
            self._dock_anchor = anchor
            return

        # For Tool windows, coordinates need to be global
        if anchor != "floating":
            px, py = self._anchor_position(parent.width(), parent.height(), anchor, margin)
            # Convert parent-relative to global coordinates
            global_pos = parent.mapToGlobal(QPoint(px, py))
            self.move(global_pos.x(), global_pos.y())
        elif position:
            # For floating, position is already global or relative to parent
            # If parent exists, convert to global
            if isinstance(parent, QWidget):
                global_pos = parent.mapToGlobal(QPoint(position[0], position[1]))
                self.move(global_pos.x(), global_pos.y())
            else:
                self.move(position[0], position[1])

        self._dock_anchor = anchor
        
        # Ensure widget stays on top after positioning
        self.raise_()

    def _anchor_position(self, parent_w: int, parent_h: int, anchor: str, margin: int) -> Tuple[int, int]:
        w = self.width()
        h = self.height()

        if anchor == "top_left":
            return margin, margin
        if anchor == "top_right":
            return max(margin, parent_w - w - margin), margin
        if anchor == "bottom_left":
            return margin, max(margin, parent_h - h - margin)
        if anchor == "bottom_right":
            return max(margin, parent_w - w - margin), max(margin, parent_h - h - margin)

        # default fallback
        return margin, margin

    # ========================================================================
    # Context menu
    # ========================================================================
    def contextMenuEvent(self, event):
        """Show a context menu for legend actions."""
        # Stop event propagation to prevent viewer's context menu from appearing
        event.accept()
        # Use unified context menu builder
        try:
            self._exec_context_menu(event.globalPos())
        except Exception:
            # Fallback to basic menu if something unexpected happens
            menu = QMenu(self)
            menu.addAction("Export Legend as PNG…", self._action_export_png)
            menu.exec(event.globalPos())

    def _action_adjust_range(self):
        """Prompt user to set new min/max and apply."""
        try:
            vmin, ok1 = QInputDialog.getDouble(
                self, "Legend Minimum", "Minimum value:", float(self.config.vmin), -1e12, 1e12, 6
            )
            if not ok1:
                logger.debug("Legend range adjust cancelled at minimum prompt")
                return
            vmax, ok2 = QInputDialog.getDouble(
                self, "Legend Maximum", "Maximum value:", float(self.config.vmax), -1e12, 1e12, 6
            )
            if not ok2:
                logger.debug("Legend range adjust cancelled at maximum prompt")
                return
            if vmax <= vmin:
                QMessageBox.warning(self, "Invalid Range", "Maximum must be greater than minimum.")
                logger.debug(
                    "Legend range adjust rejected (vmin=%s, vmax=%s)", vmin, vmax
                )
                return
            self.config.vmin = float(vmin)
            self.config.vmax = float(vmax)
            self.update()
            # Emit for host to propagate to renderer
            try:
                self.value_range_changed.emit(self.config.vmin, self.config.vmax)
            except Exception:
                pass
            logger.debug("Legend range adjusted to (%s, %s)", vmin, vmax)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to adjust range:\n{e}")
            logger.exception("Legend range adjustment failed: %s", e)

    def _action_toggle_orientation(self):
        self._orientation = "horizontal" if self._orientation == "vertical" else "vertical"
        self.update()
        try:
            self.orientation_changed.emit(self._orientation)
        except Exception:
            pass
        logger.debug("Legend orientation toggled via context menu -> %s", self._orientation)

    def _action_change_cmap(self):
        """Offer a quick selection of common colormaps."""
        try:
            common = [
                "viridis", "plasma", "inferno", "magma", "cividis",
                "hot", "coolwarm", "terrain", "Spectral", "RdYlBu"
            ]
            current = self.config.cmap_name
            cmap, ok = QInputDialog.getItem(self, "Choose Colormap", "Colormap:", common, 
                                           max(0, common.index(current)) if current in common else 0, False)
            if ok and cmap:
                logger.info(f"Legend widget: Changing colormap to '{cmap}'")
                self.config.cmap_name = cmap
                # Clear cached colormap so it will be recached with new name
                self._cached_cmap = None
                self._custom_colormap = None
                self._custom_color_samples = None
                self._cache_colormap()
                # Force repaint - invalidate entire widget to ensure color bar updates
                self.update(0, 0, self.width(), self.height())
                self.update()  # Use update() instead of repaint() to avoid recursive repaints
                # Emit signal to notify other components
                try:
                    self.colormap_changed.emit(cmap)
                    logger.info(f"Legend widget: Emitted colormap_changed signal for '{cmap}'")
                except Exception as e:
                    logger.warning(f"Failed to emit colormap_changed signal: {e}")
                logger.debug("Legend colormap changed to %s", cmap)
            else:
                logger.debug("Legend colormap selection cancelled")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to change colormap:\n{e}")
            logger.exception("Legend colormap change failed: %s", e)

    def _action_change_background(self):
        """Open a color picker to change the legend background."""
        try:
            current = getattr(self.config, "background_color", QColor(20, 20, 20, 220))
            new_color = QColorDialog.getColor(current, self, "Legend Background Color")
            if not new_color.isValid():
                logger.debug("Legend background selection cancelled")
                return
            if isinstance(current, QColor):
                new_color.setAlpha(current.alpha())
            self.config.background_color = QColor(new_color)
            self.update()
            try:
                self.background_color_changed.emit(QColor(new_color))
            except Exception:
                pass
            logger.debug(
                "Legend background color changed to rgba(%s, %s, %s, %s)",
                new_color.red(),
                new_color.green(),
                new_color.blue(),
                new_color.alpha(),
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to change background color:\n{e}")
            logger.exception("Legend background change failed: %s", e)

    def apply_style(self, style: dict):
        """Apply a consolidated legend style dictionary to the widget's config.

        Expected keys (partial): orientation, font_size, tick_count, decimals,
        background (r,g,b) in 0..1, background_opacity.
        """
        try:
            cfg = self.config
            if "orientation" in style:
                self._orientation = style.get("orientation", self._orientation)
                cfg.type = cfg.type  # keep type unchanged
            if "font_size" in style:
                cfg.font_size = int(style.get("font_size", cfg.font_size))
            if "tick_count" in style:
                cfg.tick_count = int(style.get("tick_count", cfg.tick_count))
            if "decimals" in style:
                cfg.label_decimals = int(style.get("decimals", cfg.label_decimals or 0))
            if "background" in style:
                r, g, b = style.get("background")
                cfg.background_color = QColor.fromRgbF(float(r), float(g), float(b))
            if "background_opacity" in style:
                alpha = float(style.get("background_opacity", 1.0))
                col = getattr(cfg, "background_color", QColor(20, 20, 20, 255))
                if isinstance(col, QColor):
                    new_col = QColor(col)
                    new_col.setAlphaF(alpha)
                    cfg.background_color = new_col

            # Force a repaint after applying style
            self.update()
            logger.debug("Applied legend style: %s", style)
        except Exception:
            logger.exception("Failed to apply legend style: %s", style)

    def _request_anchor_change(self, anchor: str) -> None:
        """Emit request to change docking anchor (handled by manager)."""
        self._dock_anchor = anchor
        logger.debug("User requested legend anchor change -> %s", anchor)
        try:
            self.dock_requested.emit(anchor)
        except Exception:
            pass
        if anchor == "floating":
            try:
                self.floating_position_changed.emit(self.x(), self.y())
            except Exception:
                pass

    def _action_reverse_colors(self):
        self.config.reverse = not self.config.reverse
        self._cache_colormap()
        self.update()
        try:
            self.reverse_toggled.emit(self.config.reverse)
        except Exception:
            pass
        logger.debug("Legend reverse-colors toggled -> %s", self.config.reverse)
    
    def _action_rename_category(self, category):
        """Prompt user to rename a category label."""
        try:
            current_label = self.config.category_labels.get(category, str(category))
            new_label, ok = QInputDialog.getText(
                self,
                "Rename Category Label",
                f"Enter display label for '{category}':",
                text=current_label
            )
            if ok and new_label and new_label != str(category):
                # Update local config
                self.config.category_labels[category] = new_label
                self.update()
                # Emit signal for manager to persist
                try:
                    self.category_label_changed.emit(category, new_label)
                except Exception:
                    pass
                logger.info(f"Category label renamed: {category} -> '{new_label}'")
            else:
                logger.debug("Category rename cancelled or unchanged")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to rename category:\n{e}")
            logger.exception("Category rename failed: %s", e)
    
    def _action_reset_category_label(self, category):
        """Reset a category label to its original code."""
        try:
            if category in self.config.category_labels:
                del self.config.category_labels[category]
                self.update()
                # Emit signal with empty label to indicate reset
                try:
                    self.category_label_changed.emit(category, "")
                except Exception:
                    pass
                logger.info(f"Category label reset: {category}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to reset category label:\n{e}")
            logger.exception("Category label reset failed: %s", e)

    def _exec_context_menu(self, global_pos):
        """Build and show a unified, full-featured legend context menu."""
        logger.debug("Showing legend context menu at %s", global_pos)
        menu = QMenu(self)
        
        # Range and orientation
        menu.addAction("Adjust Range…", self._action_adjust_range)
        menu.addAction(
            "Switch to Horizontal" if self._orientation == "vertical" else "Switch to Vertical",
            self._action_toggle_orientation
        )
        menu.addSeparator()
        
        # Colormap actions
        menu.addAction("Change Colormap…", self._action_change_cmap)
        reverse_action = menu.addAction("Reverse Colors", self._action_reverse_colors)
        reverse_action.setCheckable(True)
        reverse_action.setChecked(self.config.reverse)
        menu.addAction("Change Background Color…", self._action_change_background)

        # Docking options
        dock_menu = QMenu("Dock Position", self)
        dock_entries = [
            ("floating", "Floating"),
            ("top_left", "Top Left"),
            ("top_right", "Top Right"),
            ("bottom_left", "Bottom Left"),
            ("bottom_right", "Bottom Right"),
        ]
        for anchor, label in dock_entries:
            action = dock_menu.addAction(label, lambda a=anchor: self._request_anchor_change(a))
            action.setCheckable(True)
            action.setChecked(self._dock_anchor == anchor)
        menu.addMenu(dock_menu)
        
        # Category visibility submenu for discrete legends (if available)
        try:
            if getattr(self.config, 'type', None) == LegendType.DISCRETE and getattr(self.config, 'categories', None):
                visibility_menu = QMenu("Toggle Categories", self)
                for category in self.config.categories:
                    cat_action = QAction(str(category), self)
                    cat_action.setCheckable(True)
                    cat_action.setChecked(self._category_visible.get(category, True))
                    cat_action.toggled.connect(lambda checked, cat=category: self.toggle_category_visibility(cat))
                    visibility_menu.addAction(cat_action)
                menu.addMenu(visibility_menu)
        except Exception:
            pass
        
        # Category label editing for discrete legends
        try:
            if getattr(self.config, 'type', None) == LegendType.DISCRETE and self._hover_category is not None:
                menu.addSeparator()
                current_label = self.config.category_labels.get(self._hover_category, str(self._hover_category))
                rename_action = menu.addAction(f"Rename '{current_label}'…", lambda: self._action_rename_category(self._hover_category))
                
                # Only show reset if there's a custom label
                if self._hover_category in self.config.category_labels:
                    reset_action = menu.addAction(f"Reset to '{self._hover_category}'", lambda: self._action_reset_category_label(self._hover_category))
        except Exception:
            pass
        
        menu.addSeparator()

        # Legend mode switch
        if getattr(self, '_multi_mode', False):
            switch_action = menu.addAction("Switch to Classic Legend", self._action_switch_to_classic)
        else:
            switch_action = menu.addAction("Switch to Multi-Legend", self._action_switch_to_multi)

        menu.addSeparator()
        
        # Export and visibility
        menu.addAction("Export Legend as PNG…", self._action_export_png)
        hide_action = QAction("Hide Legend", self)
        hide_action.triggered.connect(self.hide)
        menu.addAction(hide_action)
        
        # Show menu at provided position
        menu.exec(global_pos)

    def _action_switch_to_classic(self):
        """Switch from multi-element mode back to classic single legend."""
        self.enable_multi_mode(False)
        logger.info("Switched to classic legend mode via context menu")

    def _action_switch_to_multi(self):
        """Switch from classic mode to multi-element legend."""
        self.enable_multi_mode(True)
        logger.info("Switched to multi-legend mode via context menu")

    def _action_export_png(self):
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Legend", "legend.png", "PNG Image (*.png)"
            )
            if filename:
                # Render to pixmap and save
                pm = QPixmap(self.size())
                pm.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pm)
                self.render(painter)
                painter.end()
                pm.save(filename, "PNG")
                logger.debug("Legend exported to %s", filename)
            else:
                logger.debug("Legend export cancelled")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to export legend:\n{e}")
            logger.exception("Legend export failed: %s", e)
    
    # ========================================================================
    # Public API - Configuration
    # ========================================================================
    
    def set_continuous(
        self,
        title: str,
        vmin: float,
        vmax: float,
        cmap_name: str = "viridis",
        tick_count: int = 6,
        log_scale: bool = False,
        reverse: bool = False,
        data: Optional[np.ndarray] = None,
        color_samples: Optional[np.ndarray] = None,
    ):
        """Configure widget for continuous scalar rendering."""
        self.config.type = LegendType.CONTINUOUS
        self.config.title = title
        self.config.vmin = float(vmin)
        self.config.vmax = float(vmax)
        self.config.cmap_name = cmap_name
        self.config.tick_count = max(2, min(15, int(tick_count)))
        self.config.log_scale = bool(log_scale)
        self.config.reverse = bool(reverse)
        self._cached_data = data.copy() if data is not None else None
        
        # Always clear custom colormap/color_samples first
        self._custom_colormap = None
        self._custom_color_samples = None
        
        # Clear cached colormap so it will be recached with the new colormap name
        # This ensures that when colormap changes, the legend uses the new colormap
        self._cached_cmap = None
        
        # Only set custom colors if color_samples are explicitly provided
        if color_samples is not None:
            try:
                samples = np.array(color_samples, dtype=float)
                if samples.ndim == 2 and samples.shape[1] >= 3 and samples.shape[0] > 0:
                    if samples.shape[1] == 3:
                        rgba = np.column_stack([samples, np.ones((samples.shape[0], 1), dtype=float)])
                    else:
                        rgba = samples[:, :4]
                    rgba = np.clip(rgba, 0.0, 1.0)
                    self._custom_color_samples = rgba
                    cmap_name_safe = title or "custom"
                    self._custom_colormap = mcolors.ListedColormap(rgba, name=f"{cmap_name_safe}_lut")
                    logger.debug(
                        "LegendWidget received %s custom color samples for %s",
                        rgba.shape[0],
                        title,
                    )
                else:
                    logger.debug(
                        "LegendWidget ignoring custom color samples for %s (shape=%s)",
                        title,
                        getattr(samples, "shape", None),
                    )
            except Exception as exc:
                logger.debug("LegendWidget failed to apply custom color samples: %s", exc)
                self._custom_color_samples = None
                self._custom_colormap = None
        
        # Validate range
        if self.config.vmin >= self.config.vmax:
            self.config.vmax = self.config.vmin + 1.0
        
        # Clear empty state on real data
        self._empty_message = None
        
        # Log before caching to debug
        logger.info(f"LegendWidget.set_continuous: title='{title}', cmap_name='{cmap_name}', has_color_samples={color_samples is not None}, has_custom_colormap={self._custom_colormap is not None}")
        
        # Cache the colormap - this will use color_samples if provided, otherwise use cmap_name
        self._cache_colormap()
        
        # Verify colormap was cached correctly
        if hasattr(self, '_cached_cmap') and self._cached_cmap is not None:
            cached_name = getattr(self._cached_cmap, 'name', 'unknown')
            logger.info(f"LegendWidget.set_continuous: Colormap cached: '{cached_name}' (requested: '{cmap_name}')")
        else:
            logger.warning(f"LegendWidget.set_continuous: Colormap was not cached after set_continuous!")
        
        # Force widget update to ensure color bar updates
        # Use update() to schedule paint event (avoid repaint() to prevent recursive repaints)
        self.update()
        
        # Also invalidate the widget's region to force a full repaint
        self.update(0, 0, self.width(), self.height())
        
        logger.debug(
            "Configured continuous legend title=%s range=(%s, %s) cmap=%s reverse=%s tick_count=%s",
            title,
            self.config.vmin,
            self.config.vmax,
            cmap_name,
            reverse,
            self.config.tick_count,
        )
    
    def set_discrete(
        self,
        title: str,
        categories: List[Union[str, int, float]],
        category_colors: Optional[Dict[Union[str, int, float], Tuple[float, float, float, float]]] = None,
        auto_colors: bool = True,
        cmap_name: Optional[str] = None,
    ):
        """Configure widget for discrete categorical rendering."""
        # DEBUG: Log what the widget receives
        logger.debug(f"[LEGEND DEBUG] LegendWidget.set_discrete called:")
        logger.debug(f"[LEGEND DEBUG]   title: {title}")
        logger.debug(f"[LEGEND DEBUG]   categories: {categories[:3] if len(categories) > 3 else categories}")
        logger.debug(f"[LEGEND DEBUG]   cmap_name: {cmap_name}")
        logger.debug(f"[LEGEND DEBUG]   auto_colors: {auto_colors}")
        logger.debug(f"[LEGEND DEBUG]   category_colors is None: {category_colors is None}")
        if category_colors:
            logger.debug(f"[LEGEND DEBUG]   Received category_colors sample:")
            for cat, color in list(category_colors.items())[:3]:
                logger.debug(f"[LEGEND DEBUG]     {cat}: {color} (type: {type(color)}, len: {len(color) if isinstance(color, (tuple, list)) else 'N/A'})")
        
        self.config.type = LegendType.DISCRETE
        self.config.title = title
        self.config.categories = list(categories)
        # Clear empty state on real data
        self._empty_message = None
        
        # Generate colors if not provided
        if category_colors is None and auto_colors:
            # Use provided colormap name for discrete categories; default to 'tab20'
            cmap_obj = cm.get_cmap(cmap_name if cmap_name else "tab20")
            n = len(categories)
            colors = {}
            if n <= 1:
                samples = [0.0]
            else:
                # Evenly sample across the colormap to maximize separability
                samples = np.linspace(0.0, 1.0, n)
            for idx, cat in enumerate(categories):
                rgba = cmap_obj(samples[idx])
                colors[cat] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
            category_colors = colors
        
        if category_colors is not None:
            self.config.category_colors = dict(category_colors)
            
            # DEBUG: Log what was actually stored in config
            logger.debug(f"[LEGEND DEBUG] Stored in config.category_colors sample:")
            for cat, color in list(self.config.category_colors.items())[:3]:
                logger.debug(f"[LEGEND DEBUG]     {cat}: {color} (type: {type(color)}, len: {len(color) if isinstance(color, (tuple, list)) else 'N/A'})")
        else:
            logger.debug(f"[LEGEND DEBUG] No category_colors set, config.category_colors is empty")
        
        # Initialize visibility
        for cat in categories:
            if cat not in self._category_visible:
                self._category_visible[cat] = True
        
        # Force widget update and repaint to ensure colors are displayed
        self.update()
        self.repaint()
        # Also invalidate the widget's region to force a full repaint
        self.update(0, 0, self.width(), self.height())
        
        logger.debug(
            "Configured discrete legend title=%s categories=%s auto_colors=%s cmap=%s",
            title,
            len(categories),
            auto_colors and category_colors is None,
            cmap_name,
        )
    
    def set_orientation(self, orientation: str):
        """Set legend orientation: 'vertical' or 'horizontal'."""
        self._orientation = "horizontal" if str(orientation).lower().startswith("h") else "vertical"
        logger.debug("Legend orientation set to %s", self._orientation)
        self.update()
    
    def set_empty_state(self, message: str = "No data"):
        """Show an empty-state legend with a message; hides bars and categories."""
        self._empty_message = message
        logger.debug("Legend empty state: %s", message)
        self.update()

    def clear_empty_state(self):
        """Clear the empty-state; no effect if data not set until next update call."""
        self._empty_message = None
        self.update()

    def set_theme(
        self,
        background: Optional[QColor] = None,
        text: Optional[QColor] = None,
        border: Optional[QColor] = None,
        font_size: Optional[int] = None
    ):
        """Update visual theme."""
        if background is not None:
            self.config.background_color = background
        if text is not None:
            self.config.text_color = text
        if border is not None:
            self.config.border_color = border
        if font_size is not None:
            self.config.font_size = max(8, min(24, int(font_size)))
        self.update()
        logger.debug(
            "Legend theme updated background=%s text=%s border=%s font_size=%s",
            background,
            text,
            border,
            font_size,
        )

    def set_subtitle(self, subtitle: str = ""):
        """Set optional subtitle (e.g., units like '% Fe')."""
        self.config.subtitle = subtitle or ""
        self.update()
    
    def set_markers(self, values: List[float], show_mean: bool = False, show_median: bool = False):
        """Set additional marker values to display on continuous legend."""
        self.config.markers = list(values)
        self.config.show_mean = show_mean
        self.config.show_median = show_median
        self.update()
        logger.debug(
            "Legend markers updated count=%s show_mean=%s show_median=%s",
            len(values),
            show_mean,
            show_median,
        )
    
    def toggle_category_visibility(self, category: Union[str, int, float]):
        """Toggle visibility of a category (discrete mode)."""
        if category in self._category_visible:
            self._category_visible[category] = not self._category_visible[category]
            self.category_toggled.emit(category, self._category_visible[category])
            self.update()
            logger.debug(
                "Category visibility toggled %s -> %s",
                category,
                self._category_visible[category],
            )
    
    def is_category_visible(self, category: Union[str, int, float]) -> bool:
        """Check if a category is visible."""
        return self._category_visible.get(category, True)

    # ========================================================================
    # Multi-Element Mode API
    # ========================================================================

    def enable_multi_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable multi-element mode.

        In multi-mode, the legend can display multiple elements with
        add/remove controls. In single mode, it displays a single legend.
        """
        logger.info(f"enable_multi_mode called: enabled={enabled}, current={getattr(self, '_multi_mode', 'NOT_SET')}")

        # Ensure multi-mode attributes are initialized
        if not hasattr(self, '_multi_mode'):
            self._multi_mode = False
        if not hasattr(self, '_elements'):
            self._elements = []
        if not hasattr(self, '_button_rects'):
            self._button_rects = {}
        if not hasattr(self, '_element_rects'):
            self._element_rects = {}

        if self._multi_mode == enabled:
            return

        self._multi_mode = enabled
        logger.info(f"Multi-mode set to: {self._multi_mode}")
        self.mode_changed.emit(enabled)
        if enabled:
            # Convert current single config to an element if present
            if self.config.title and (self.config.vmin != self.config.vmax or self.config.categories):
                elem = self._config_to_element(self.config)
                if elem and elem.id not in [e.id for e in self._elements]:
                    self._elements.append(elem)
            logger.debug("Multi-mode enabled with %d elements", len(self._elements))
        else:
            logger.debug("Multi-mode disabled")

        self.update()

    def is_multi_mode(self) -> bool:
        """Check if multi-element mode is active."""
        return self._multi_mode

    def add_element(self, element: LegendElement) -> None:
        """
        Add a legend element.

        If an element with the same ID exists, it will be updated.
        """
        # Enable multi-mode if not already
        if not self._multi_mode:
            self.enable_multi_mode(True)

        # Remove existing element with same id
        self._elements = [e for e in self._elements if e.id != element.id]
        self._elements.append(element)

        self.element_added.emit(element.id)
        logger.debug("Added element: %s (%s)", element.id, element.element_type.value)
        self.update()

    def remove_element(self, element_id: str) -> bool:
        """
        Remove an element by ID.

        Returns True if element was found and removed, False otherwise.
        """
        original_len = len(self._elements)
        self._elements = [e for e in self._elements if e.id != element_id]

        if len(self._elements) < original_len:
            self.element_removed.emit(element_id)
            logger.debug("Removed element: %s", element_id)
            self.update()
            return True
        return False

    def get_element(self, element_id: str) -> Optional[LegendElement]:
        """Get element by ID."""
        return next((e for e in self._elements if e.id == element_id), None)

    def get_elements(self) -> List[LegendElement]:
        """Get all elements."""
        return list(self._elements)

    def clear_elements(self) -> None:
        """Remove all elements."""
        self._elements.clear()
        logger.debug("Cleared all elements")
        self.update()

    def set_element_visibility(self, element_id: str, visible: bool) -> bool:
        """Set visibility for an element."""
        elem = self.get_element(element_id)
        if elem:
            elem.visible = visible
            self.element_visibility_changed.emit(element_id, visible)
            self.update()
            return True
        return False

    def _config_to_element(self, config: LegendConfig) -> Optional[LegendElement]:
        """Convert a single LegendConfig to a LegendElement."""
        if not config.title:
            return None

        if config.type == LegendType.CONTINUOUS:
            return LegendElement.create_continuous(
                layer="",
                property_name=config.title,
                title=config.title,
                vmin=config.vmin,
                vmax=config.vmax,
                cmap_name=config.cmap_name,
                reverse=config.reverse,
            )
        else:
            return LegendElement.create_discrete(
                layer="",
                property_name=config.title,
                title=config.title,
                categories=config.categories,
                category_colors=config.category_colors,
            )

    # ========================================================================
    # Rendering - Core Paint Event
    # ========================================================================
    
    def paintEvent(self, event):
        """Main rendering entry point. Errors are handled gracefully to avoid app crashes."""
        qp = None
        try:
            qp = QPainter(self)
            qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            qp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            qp.setCompositionMode(QPainter.CompositionMode_Source)

            rect = self.rect()

            # Draw background with rounded corners
            self._draw_background(qp, rect)

            # Multi-element mode rendering
            multi_mode = getattr(self, '_multi_mode', False)
            if multi_mode:
                logger.debug("paintEvent: Drawing in multi-mode")
                self._draw_multi_mode(qp, rect)
                # Draw resize handle and return
                self._draw_resize_handle(qp, rect)
                # End painter here and set to None so finally block doesn't double-end
                qp.end()
                qp = None
                return

            # Draw property toggles (checkboxes/pills) for each layer property
            if hasattr(self, '_layer_properties') and self._layer_properties:
                x = self._padding
                y = rect.height() - self._padding - 24
                for prop in self._layer_properties:
                    toggled = getattr(self, '_property_toggled', {}).get(prop, True)
                    pill_color = QColor(60, 180, 120) if toggled else QColor(180, 60, 60)
                    pill_rect = QRect(x, y, 80, 22)
                    qp.setBrush(pill_color)
                    qp.setPen(Qt.PenStyle.NoPen)
                    qp.drawRoundedRect(pill_rect, 10, 10)
                    qp.setPen(QColor(255, 255, 255))
                    qp.setFont(QFont(self.config.font_family, 11))
                    qp.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, prop)
                    x += 90

            # Adaptive theme/contrast: auto-adjust text/icon color based on background
            bg = self.config.background_color
            brightness = (bg.red() * 0.299 + bg.green() * 0.587 + bg.blue() * 0.114)
            text_color = QColor(20, 20, 20) if brightness > 180 else QColor(245, 245, 245)
            qp.setPen(text_color)

            # Draw resize handle (bottom-right corner)
            handle_size = getattr(self, '_resize_handle_size', 12)
            handle_rect = QRect(self.width() - handle_size, self.height() - handle_size, handle_size, handle_size)
            qp.setBrush(QColor(180, 180, 180, 180))
            qp.setPen(Qt.PenStyle.NoPen)
            qp.drawRect(handle_rect)

            # Draw legend content based on type
            if self._empty_message:
                qp.setPen(QPen(self.config.text_color))
                qp.setFont(QFont(self.config.font_family, self.config.font_size))
                qp.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._empty_message)
            elif self.config.type == LegendType.CONTINUOUS:
                # Ensure colormap is cached before drawing
                if not hasattr(self, "_cached_cmap") or self._cached_cmap is None:
                    try:
                        self._cache_colormap()
                    except Exception as e:
                        logger.warning(f"Could not cache colormap in paintEvent: {e}")
                # Only draw if we have valid range and colormap
                if self.config.vmin < self.config.vmax and (hasattr(self, "_cached_cmap") and self._cached_cmap is not None):
                    self._draw_continuous(qp, rect)
                else:
                    # Show message if data is invalid
                    qp.setPen(QPen(self.config.text_color))
                    qp.setFont(QFont(self.config.font_family, self.config.font_size))
                    status_msg = "No colormap" if not (hasattr(self, "_cached_cmap") and self._cached_cmap is not None) else "Invalid range"
                    qp.drawText(rect, Qt.AlignmentFlag.AlignCenter, status_msg)
            elif self.config.type == LegendType.DISCRETE:
                self._draw_discrete(qp, rect)
            else:
                    # Empty state
                    qp.setPen(QPen(self.config.text_color))
                    qp.setFont(QFont(self.config.font_family, self.config.font_size))
                    qp.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No legend data")

            # Draw resize handle
            self._draw_resize_handle(qp, rect)
        except Exception as e:
            # Log and show a minimal error indicator inside the widget
            import logging
            logging.getLogger(__name__).exception("Legend paint error: %s", e)
            if qp is not None:
                try:
                    qp.setPen(QPen(QColor(200, 60, 60)))
                    qp.setFont(QFont(self.config.font_family, 10))
                    qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Legend render error")
                except Exception:
                    # If drawing error message fails, just skip it
                    pass
        finally:
            if qp is not None:
                qp.end()
    
    def _draw_background(self, qp: QPainter, rect: QRect):
        """Draw the legend background card — fully opaque, no double blending."""
        # Simplified: force an opaque rounded rect using the configured background color
        base_color = QColor(self.config.background_color)
        top_color = QColor(base_color)
        top_color.setAlpha(255)

        qp.save()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        qp.setBrush(top_color)
        qp.setPen(Qt.PenStyle.NoPen)
        # Use QRectF to support floating point radii
        qp.drawRoundedRect(QRectF(self.rect()), float(getattr(self.config, 'card_corner_radius', 10.0)), float(getattr(self.config, 'card_corner_radius', 10.0)))
        qp.restore()

    # ========================================================================
    # Rendering - Multi-Element Mode
    # ========================================================================

    def _draw_multi_mode(self, qp: QPainter, rect: QRect):
        """Render the legend in multi-element mode with toolbar and stacked elements."""
        self._button_rects.clear()
        self._element_rects.clear()

        padding = self._padding
        y_offset = padding

        # Draw toolbar
        toolbar_rect = QRect(padding, y_offset, rect.width() - 2 * padding, self._toolbar_height)
        self._draw_multi_toolbar(qp, toolbar_rect)
        y_offset += self._toolbar_height + 8

        # Draw elements
        if not self._elements:
            # Empty state
            qp.setPen(QPen(QColor(150, 150, 150)))
            qp.setFont(QFont(self.config.font_family, 11))
            empty_rect = QRect(padding, y_offset, rect.width() - 2 * padding, 40)
            qp.drawText(empty_rect, Qt.AlignmentFlag.AlignCenter, "Click + to add legend elements")
        else:
            for element in self._elements:
                if not element.visible:
                    # Draw collapsed element (just header)
                    header_height = self._element_header_height
                    element_rect = QRect(padding, y_offset, rect.width() - 2 * padding, header_height)
                    self._element_rects[element.id] = element_rect
                    self._draw_element_header(qp, element_rect, element)
                    y_offset += header_height + 4
                else:
                    # Draw full element
                    content_height = self._get_element_content_height(element)
                    total_height = self._element_header_height + content_height
                    element_rect = QRect(padding, y_offset, rect.width() - 2 * padding, total_height)
                    self._element_rects[element.id] = element_rect

                    # Draw element background
                    qp.save()
                    qp.setBrush(QColor(30, 30, 35))
                    qp.setPen(QPen(QColor(70, 70, 75), 1))
                    qp.drawRoundedRect(QRectF(element_rect), 6, 6)
                    qp.restore()

                    # Draw header
                    header_rect = QRect(element_rect.x(), element_rect.y(),
                                       element_rect.width(), self._element_header_height)
                    self._draw_element_header(qp, header_rect, element)

                    # Draw content
                    content_rect = QRect(element_rect.x(), element_rect.y() + self._element_header_height,
                                        element_rect.width(), content_height)
                    self._draw_element_content(qp, content_rect, element)

                    y_offset += total_height + 8

    def _draw_multi_toolbar(self, qp: QPainter, rect: QRect):
        """Draw the multi-mode toolbar with [+] button."""
        # Toolbar background
        qp.save()
        qp.setBrush(QColor(40, 40, 45))
        qp.setPen(Qt.PenStyle.NoPen)
        qp.drawRoundedRect(QRectF(rect), 8, 8)
        qp.restore()

        # Title
        qp.setPen(QPen(QColor(245, 245, 245)))
        title_font = QFont(self.config.font_family, 12)
        title_font.setBold(True)
        qp.setFont(title_font)
        title_rect = QRect(rect.x() + 12, rect.y(), 100, rect.height())
        qp.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, "LEGEND")

        # [+] Add button
        btn_size = 24
        btn_x = rect.right() - btn_size - 8
        btn_y = rect.y() + (rect.height() - btn_size) // 2
        add_btn_rect = QRect(btn_x, btn_y, btn_size, btn_size)
        self._button_rects['add'] = add_btn_rect

        # Button style
        is_hovered = self._hover_button == 'add'
        btn_color = QColor(80, 80, 85) if is_hovered else QColor(60, 60, 65)
        qp.save()
        qp.setBrush(btn_color)
        qp.setPen(QPen(QColor(100, 100, 105), 1))
        qp.drawRoundedRect(QRectF(add_btn_rect), 4, 4)
        qp.restore()

        # Plus sign
        qp.setPen(QPen(QColor(200, 200, 200), 2))
        center_x = add_btn_rect.center().x()
        center_y = add_btn_rect.center().y()
        qp.drawLine(center_x - 5, center_y, center_x + 5, center_y)
        qp.drawLine(center_x, center_y - 5, center_x, center_y + 5)

    def _draw_element_header(self, qp: QPainter, rect: QRect, element: LegendElement):
        """Draw the header row for an element (visibility, title, remove)."""
        # Header background
        qp.save()
        qp.setBrush(QColor(50, 50, 55))
        qp.setPen(Qt.PenStyle.NoPen)
        # Only round top corners if element is visible (content below)
        if element.visible:
            path = QPainterPath()
            path.moveTo(rect.left() + 6, rect.bottom())
            path.lineTo(rect.left(), rect.bottom())
            path.lineTo(rect.left(), rect.top() + 6)
            path.arcTo(QRectF(rect.left(), rect.top(), 12, 12), 180, -90)
            path.lineTo(rect.right() - 6, rect.top())
            path.arcTo(QRectF(rect.right() - 12, rect.top(), 12, 12), 90, -90)
            path.lineTo(rect.right(), rect.bottom())
            path.lineTo(rect.left() + 6, rect.bottom())
            qp.fillPath(path, QBrush(QColor(50, 50, 55)))
        else:
            qp.drawRoundedRect(QRectF(rect), 6, 6)
        qp.restore()

        btn_size = 20
        padding = 6

        # Visibility toggle button
        vis_btn_rect = QRect(rect.x() + padding, rect.y() + (rect.height() - btn_size) // 2,
                            btn_size, btn_size)
        self._button_rects[f'vis_{element.id}'] = vis_btn_rect

        is_vis_hovered = self._hover_button == f'vis_{element.id}'
        vis_color = QColor(60, 120, 80) if element.visible else QColor(80, 60, 60)
        if is_vis_hovered:
            vis_color = vis_color.lighter(120)
        qp.save()
        qp.setBrush(vis_color)
        qp.setPen(QPen(QColor(80, 80, 85), 1))
        qp.drawRoundedRect(QRectF(vis_btn_rect), 4, 4)
        qp.restore()

        # Eye icon (simple)
        qp.setPen(QPen(QColor(220, 220, 220), 1.5))
        eye_text = "O" if element.visible else "-"
        qp.setFont(QFont(self.config.font_family, 10, QFont.Weight.Bold))
        qp.drawText(vis_btn_rect, Qt.AlignmentFlag.AlignCenter, eye_text)

        # Title
        title_x = vis_btn_rect.right() + 8
        title_rect = QRect(title_x, rect.y(), rect.width() - title_x - btn_size - padding * 2, rect.height())
        qp.setPen(QPen(QColor(230, 230, 230)))
        qp.setFont(QFont(self.config.font_family, 11, QFont.Weight.DemiBold))
        # Truncate title if too long
        fm = QFontMetrics(qp.font())
        title_text = fm.elidedText(element.title, Qt.TextElideMode.ElideRight, title_rect.width())
        qp.drawText(title_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title_text)

        # Remove button
        rem_btn_rect = QRect(rect.right() - btn_size - padding, rect.y() + (rect.height() - btn_size) // 2,
                            btn_size, btn_size)
        self._button_rects[f'rem_{element.id}'] = rem_btn_rect

        is_rem_hovered = self._hover_button == f'rem_{element.id}'
        rem_color = QColor(150, 60, 60) if is_rem_hovered else QColor(70, 50, 50)
        qp.save()
        qp.setBrush(rem_color)
        qp.setPen(QPen(QColor(100, 70, 70), 1))
        qp.drawRoundedRect(QRectF(rem_btn_rect), 4, 4)
        qp.restore()

        # X icon
        qp.setPen(QPen(QColor(220, 220, 220), 1.5))
        qp.setFont(QFont(self.config.font_family, 10, QFont.Weight.Bold))
        qp.drawText(rem_btn_rect, Qt.AlignmentFlag.AlignCenter, "x")

    def _draw_element_content(self, qp: QPainter, rect: QRect, element: LegendElement):
        """Draw the content area for an element (gradient or categories)."""
        if element.element_type == LegendElementType.CONTINUOUS:
            self._draw_element_continuous(qp, rect, element)
        else:
            self._draw_element_discrete(qp, rect, element)

    def _draw_element_continuous(self, qp: QPainter, rect: QRect, element: LegendElement):
        """Draw a continuous gradient element."""
        if element.vmin is None or element.vmax is None or element.vmin >= element.vmax:
            return

        padding = 8
        bar_width = 18
        bar_left = rect.x() + padding
        bar_top = rect.y() + padding
        bar_height = rect.height() - 2 * padding
        bar_rect = QRectF(bar_left, bar_top, bar_width, bar_height)

        # Get colormap
        try:
            cmap = cm.get_cmap(element.cmap_name or 'viridis')
        except Exception:
            cmap = cm.get_cmap('viridis')

        # Draw gradient bar
        grad = QLinearGradient()
        grad.setStart(bar_rect.bottomLeft())
        grad.setFinalStop(bar_rect.topLeft())

        for i in range(32):
            n = i / 31
            if element.reverse:
                n = 1.0 - n
            try:
                rgba = cmap(n)
                color = QColor.fromRgbF(float(rgba[0]), float(rgba[1]), float(rgba[2]), 1.0)
            except Exception:
                color = QColor(128, 128, 128)
            grad.setColorAt(i / 31, color)

        qp.save()
        qp.setBrush(QBrush(grad))
        qp.setPen(QPen(QColor(60, 60, 60), 1))
        qp.drawRoundedRect(bar_rect, 4, 4)
        qp.restore()

        # Draw tick labels
        font = QFont(self.config.font_family, 10)
        qp.setFont(font)
        fm = QFontMetrics(font)
        qp.setPen(QPen(QColor(220, 220, 220)))

        tick_count = min(element.tick_count, 5)
        for i in range(tick_count):
            n = i / (tick_count - 1) if tick_count > 1 else 0.5
            if element.reverse:
                n = 1.0 - n
            val = element.vmin + n * (element.vmax - element.vmin)
            y = bar_rect.top() + (1.0 - n) * bar_rect.height()

            # Format value
            if abs(val) >= 1000:
                label_text = f"{val:.0f}"
            elif abs(val) >= 10:
                label_text = f"{val:.1f}"
            else:
                label_text = f"{val:.2f}"

            label_x = bar_rect.right() + 8
            label_y = y + fm.ascent() / 3
            qp.drawText(int(label_x), int(label_y), label_text)

    def _draw_element_discrete(self, qp: QPainter, rect: QRect, element: LegendElement):
        """Draw a discrete category element."""
        if not element.categories:
            return

        padding = 8
        box_size = 14
        row_height = 20
        row_spacing = 2

        font = QFont(self.config.font_family, 10)
        qp.setFont(font)
        fm = QFontMetrics(font)

        y = rect.y() + padding
        x = rect.x() + padding

        # Limit visible categories
        visible_categories = element.categories[:8]

        for category in visible_categories:
            cat_visible = element.category_visible.get(category, True)

            # Color box
            box_rect = QRectF(x, y, box_size, box_size)
            color = element.category_colors.get(category, (0.5, 0.5, 0.5, 1.0))

            if isinstance(color, (tuple, list)) and len(color) >= 3:
                r, g, b = float(color[0]), float(color[1]), float(color[2])
                a = float(color[3]) if len(color) > 3 else 1.0
                if r > 1.0 or g > 1.0 or b > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                swatch_color = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            else:
                swatch_color = QColor(128, 128, 128, 255)

            if not cat_visible:
                swatch_color.setAlpha(80)

            qp.save()
            qp.setBrush(QBrush(swatch_color))
            qp.setPen(QPen(QColor(80, 80, 80), 1))
            qp.drawRoundedRect(box_rect, 3, 3)
            qp.restore()

            # Label
            label_text = str(element.category_labels.get(category, category))
            if len(label_text) > 20:
                label_text = label_text[:18] + ".."

            label_x = x + box_size + 6
            label_y = y + fm.ascent() - 1

            if cat_visible:
                qp.setPen(QPen(QColor(220, 220, 220)))
            else:
                qp.setPen(QPen(QColor(100, 100, 100)))

            qp.drawText(int(label_x), int(label_y), label_text)

            y += row_height + row_spacing

        # Show "+N more" if truncated
        if len(element.categories) > 8:
            more_count = len(element.categories) - 8
            qp.setPen(QPen(QColor(150, 150, 150)))
            qp.drawText(int(x), int(y + fm.ascent()), f"+{more_count} more...")

    def _get_element_content_height(self, element: LegendElement) -> int:
        """Calculate the content height for an element."""
        if element.element_type == LegendElementType.CONTINUOUS:
            return 80  # Fixed height for gradient
        else:
            # Height based on category count
            count = min(len(element.categories), 8)
            height = 8 + count * 22 + 8
            if len(element.categories) > 8:
                height += 20  # For "+N more" text
            return max(50, height)

    # ========================================================================
    # Rendering - Continuous Mode
    # ========================================================================
    
    def _draw_continuous(self, qp: QPainter, rect: QRect):
        """Render continuous scalar legend."""
        if self.config.vmin >= self.config.vmax:
            return
        
        font = QFont(self.config.font_family, self.config.font_size)
        if getattr(self.config, 'title_bold', True):
            font.setWeight(QFont.Weight.DemiBold)
        qp.setFont(font)
        fm = QFontMetrics(font)
        
        if self._orientation == "vertical":
            self._draw_continuous_vertical(qp, rect, fm)
        else:
            self._draw_continuous_horizontal(qp, rect, fm)
    
    def _draw_continuous_vertical(self, qp: QPainter, rect: QRect, fm: QFontMetrics):
        """Render vertical continuous legend."""
        padding = self._padding
        title_height = fm.height() + 4
        bar_width = self._bar_width
        
        # Title
        qp.setPen(QPen(self.config.text_color))
        title_rect = QRectF(padding, padding, rect.width() - 2 * padding, title_height)
        qp.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.config.title)
        # Optional subtitle (e.g., units)
        if getattr(self.config, 'subtitle', ''):
            sub_font = QFont(self.config.font_family, max(9, self.config.font_size - 2))
            qp.setFont(sub_font)
            sub_height = QFontMetrics(sub_font).height()
            sub_rect = QRectF(padding, padding + title_height - 2, rect.width() - 2 * padding, sub_height)
            qp.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.config.subtitle)
            qp.setFont(QFont(self.config.font_family, self.config.font_size))
            title_height += sub_height
        
        # Color bar area
        bar_top = padding + title_height + 8
        bar_left = padding
        bar_height = rect.height() - bar_top - padding - 40  # Space for ticks
        bar_rect = QRectF(bar_left, bar_top, bar_width, bar_height)
        
        # Draw gradient
        self._draw_gradient_bar(qp, bar_rect, vertical=True)
        
        # Draw ticks and labels
        tick_values = self._compute_tick_values()
        qp.setPen(QPen(getattr(self.config, 'tick_color', self.config.text_color), float(getattr(self.config, 'tick_width', 1.5))))

        # Collect candidate ticks (sorted by y)
        candidates = []
        for val in tick_values:
            n = self._normalize_value(val)
            if self.config.reverse:
                n = 1.0 - n
            y = bar_rect.top() + (1.0 - n) * bar_rect.height()
            label_text = self._format_value(val)
            candidates.append((y, label_text))
        candidates.sort(key=lambda t: t[0])

        min_gap = fm.height() + 2
        selected = []
        last_y = -1e9
        for i, (y, label_text) in enumerate(candidates):
            if i == 0 or (y - last_y) >= min_gap or i == len(candidates) - 1:
                selected.append((y, label_text))
                last_y = y

        for y, label_text in selected:
            tick_x_end = bar_rect.right() + self._tick_spacing
            qp.drawLine(int(bar_rect.right()), int(y), int(tick_x_end), int(y))
            qp.drawText(int(tick_x_end + 4), int(y + fm.ascent() / 2), label_text)
        
        # Draw hover marker
        if self._hover_value is not None:
            n = self._normalize_value(self._hover_value)
            if self.config.reverse:
                n = 1.0 - n
            y = bar_rect.top() + (1.0 - n) * bar_rect.height()
            
            # Highlight line
            qp.setPen(QPen(QColor(255, 200, 0), 2))
            qp.drawLine(
                int(bar_rect.left() - 2),
                int(y),
                int(bar_rect.right() + 2),
                int(y)
            )
            
            # Value label
            hover_text = f"{self._hover_value:.3f}"
            hover_width = fm.horizontalAdvance(hover_text)
            hover_x = bar_rect.right() + self._tick_spacing + 4
            hover_y = int(y - fm.height() / 2)
            
            # Background for label
            label_bg = QRectF(
                hover_x - 2,
                hover_y - 2,
                hover_width + 4,
                fm.height() + 4
            )
            qp.fillRect(label_bg, QColor(0, 0, 0, 200))
            
            qp.setPen(QPen(QColor(255, 255, 255)))
            qp.drawText(int(hover_x), int(hover_y + fm.ascent()), hover_text)
        
        # Draw additional markers
        markers = self.config.get_marker_values(self._cached_data)
        for marker_val in markers:
            if self.config.vmin <= marker_val <= self.config.vmax:
                n = self._normalize_value(marker_val)
                if self.config.reverse:
                    n = 1.0 - n
                y = bar_rect.top() + (1.0 - n) * bar_rect.height()
                
                # Dashed marker line
                pen = QPen(QColor(255, 100, 100), 1.5)
                pen.setStyle(Qt.PenStyle.DashLine)
                qp.setPen(pen)
                qp.drawLine(
                    int(bar_rect.left() - 4),
                    int(y),
                    int(bar_rect.right() + 4),
                    int(y)
                )
    
    def _draw_continuous_horizontal(self, qp: QPainter, rect: QRect, fm: QFontMetrics):
        """Render horizontal continuous legend."""
        padding = self._padding
        title_height = fm.height() + 4
        bar_height = self._bar_width
        
        # Title
        qp.setPen(QPen(self.config.text_color))
        title_rect = QRectF(padding, padding, rect.width() - 2 * padding, title_height)
        qp.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.config.title)
        if getattr(self.config, 'subtitle', ''):
            sub_font = QFont(self.config.font_family, max(9, self.config.font_size - 2))
            qp.setFont(sub_font)
            sub_height = QFontMetrics(sub_font).height()
            sub_rect = QRectF(padding, padding + title_height - 2, rect.width() - 2 * padding, sub_height)
            qp.drawText(sub_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.config.subtitle)
            qp.setFont(QFont(self.config.font_family, self.config.font_size))
            title_height += sub_height
        
        # Color bar area
        bar_top = padding + title_height + 8
        bar_left = padding + 40  # Space for labels on left
        bar_width = rect.width() - bar_left - padding - 40  # Space for labels on right
        bar_rect = QRectF(bar_left, bar_top, bar_width, bar_height)
        
        # Draw gradient
        self._draw_gradient_bar(qp, bar_rect, vertical=False)
        
        # Draw ticks and labels
        tick_values = self._compute_tick_values()
        qp.setPen(QPen(getattr(self.config, 'tick_color', self.config.text_color), float(getattr(self.config, 'tick_width', 1.5))))

        candidates = []
        for val in tick_values:
            n = self._normalize_value(val)
            if self.config.reverse:
                n = 1.0 - n
            x = bar_rect.left() + n * bar_rect.width()
            label_text = self._format_value(val)
            width = fm.horizontalAdvance(label_text)
            candidates.append((x, label_text, width))
        candidates.sort(key=lambda t: t[0])

        selected = []
        last_right = -1e9
        for i, (x, label_text, width) in enumerate(candidates):
            left = x - width / 2
            right = x + width / 2
            if i == 0 or (left - last_right) >= 8 or i == len(candidates) - 1:
                selected.append((x, label_text, width))
                last_right = right

        for x, label_text, width in selected:
            tick_y_end = bar_rect.bottom() + self._tick_spacing
            qp.drawLine(int(x), int(bar_rect.bottom()), int(x), int(tick_y_end))
            label_x = int(x - width / 2)
            label_y = int(tick_y_end + fm.height() + 2)
            qp.drawText(label_x, label_y, label_text)
        
        # Draw hover marker
        if self._hover_value is not None:
            n = self._normalize_value(self._hover_value)
            if self.config.reverse:
                n = 1.0 - n
            x = bar_rect.left() + n * bar_rect.width()
            
            # Highlight line
            qp.setPen(QPen(QColor(255, 200, 0), 2))
            qp.drawLine(
                int(x),
                int(bar_rect.top() - 2),
                int(x),
                int(bar_rect.bottom() + 2)
            )
            
            # Value label above
            hover_text = f"{self._hover_value:.3f}"
            hover_width = fm.horizontalAdvance(hover_text)
            hover_x = int(x - hover_width / 2)
            hover_y = int(bar_rect.top() - fm.height() - 4)
            
            # Background for label
            label_bg = QRectF(
                hover_x - 2,
                hover_y - 2,
                hover_width + 4,
                fm.height() + 4
            )
            qp.fillRect(label_bg, QColor(0, 0, 0, 200))
            
            qp.setPen(QPen(QColor(255, 255, 255)))
            qp.drawText(hover_x, hover_y + fm.ascent(), hover_text)
    
    def _draw_gradient_bar(self, qp, rect: QRectF, vertical: bool = True):
        """Render continuous gradient bar with perfect scaling.

        This implementation builds a QLinearGradient from samples taken from
        ``self._cached_cmap`` and forces full opacity for every stop to avoid
        faded / transparent artefacts when the source LUT provides 0..255
        unsigned byte values or when the colormap callable returns different
        shapes (QColor, array, tuple).
        """
        # Always check if cached colormap matches config - if not, recache
        config_cmap_name = getattr(self.config, 'cmap_name', 'viridis')
        if hasattr(self, "_cached_cmap") and self._cached_cmap is not None:
            cached_name = getattr(self._cached_cmap, 'name', 'unknown')
            # Check if cached colormap name matches config (accounting for _cached suffix)
            cached_base_name = cached_name.replace('_cached', '').replace('_lut', '')
            config_base_name = config_cmap_name.replace('_custom', '')
            
            if cached_base_name != config_base_name:
                logger.info(f"_draw_gradient_bar: Colormap mismatch detected! Cached: '{cached_base_name}', Config: '{config_base_name}'. Recaching...")
                # Clear and recache
                self._cached_cmap = None
                self._custom_colormap = None
                self._custom_color_samples = None
                self._cache_colormap()
        
        # Log which colormap is being used for debugging
        if hasattr(self, "_cached_cmap") and self._cached_cmap is not None:
            cached_name = getattr(self._cached_cmap, 'name', 'unknown')
            logger.debug(f"_draw_gradient_bar: using cached colormap '{cached_name}' (config says '{config_cmap_name}')")
        
        if not hasattr(self, "_cached_cmap") or self._cached_cmap is None:
            # Try to cache colormap if it's missing
            try:
                logger.debug("Colormap not cached in _draw_gradient_bar, caching now")
                self._cache_colormap()
                if not hasattr(self, "_cached_cmap") or self._cached_cmap is None:
                    logger.warning("Failed to cache colormap, gradient bar will not be drawn")
                    return
            except Exception as e:
                logger.error(f"Error caching colormap in _draw_gradient_bar: {e}", exc_info=True)
                return

        # Determine orientation
        is_vertical = self._orientation == "vertical" if vertical is None else vertical
        grad = QLinearGradient()
        if is_vertical:
            grad.setStart(rect.bottomLeft())
            grad.setFinalStop(rect.topLeft())
        else:
            grad.setStart(rect.topLeft())
            grad.setFinalStop(rect.topRight())

        # Build gradient stops directly from Matplotlib colormap
        # Use more steps for smoother gradient (256 is good)
        steps = 256
        for i in range(steps):
            n = i / (steps - 1)
            try:
                # Try calling the colormap directly
                rgba = self._cached_cmap(n)
                # Ensure it's a tuple/array with 4 elements (RGBA)
                if isinstance(rgba, (list, tuple, np.ndarray)):
                    rgba = np.asarray(rgba, dtype=float)
                    if rgba.size == 3:
                        rgba = np.append(rgba, 1.0)  # Add alpha if missing
                    elif rgba.size != 4:
                        rgba = np.array([1.0, 1.0, 1.0, 1.0])  # Fallback white
                else:
                    rgba = np.array([1.0, 1.0, 1.0, 1.0])
            except Exception as e:
                try:
                    # Try with array input
                    rgba = np.asarray(self._cached_cmap(np.array([n])))[0]
                    if rgba.size == 3:
                        rgba = np.append(rgba, 1.0)
                    elif rgba.size != 4:
                        rgba = np.array([1.0, 1.0, 1.0, 1.0])
                except Exception:
                    logger.debug(f"Failed to get color at position {n} from colormap, using white")
                    rgba = np.array([1.0, 1.0, 1.0, 1.0])

            # Normalize to float 0..1 and ensure length 4
            try:
                arr = np.asarray(rgba, dtype=float)
                if arr.max() > 1.01:
                    arr = arr / 255.0
            except Exception:
                arr = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

            if arr.size < 3:
                arr = np.array([float(arr.flat[0]), float(arr.flat[0]), float(arr.flat[0]), 1.0], dtype=float)
            if arr.size == 3:
                arr = np.append(arr, 1.0)

            color = QColor.fromRgbF(float(arr[0]), float(arr[1]), float(arr[2]), 1.0)
            # ensure fully-opaque in case Qt mixes premultiplied values
            color.setAlpha(255)
            grad.setColorAt(n, color)

        # Draw with rounded edges and no outline
        qp.save()
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setBrush(QBrush(grad))
        qp.setPen(Qt.PenStyle.NoPen)
        qp.drawRoundedRect(rect, self.config.bar_corner_radius, self.config.bar_corner_radius)
        qp.restore()
    
    # ========================================================================
    # Rendering - Discrete Mode
    # ========================================================================
    
    def _draw_discrete(self, qp: QPainter, rect: QRect):
        """Render discrete categorical legend."""
        if not self.config.categories:
            return
        
        font = QFont(self.config.font_family, self.config.font_size)
        if getattr(self.config, 'title_bold', True):
            font.setWeight(QFont.Weight.DemiBold)
        qp.setFont(font)
        fm = QFontMetrics(font)
        
        padding = self._padding
        title_height = fm.height() + 4
        box_size = self._category_box_size
        row_height = max(box_size + 4, fm.height() + 4)
        row_spacing = 2
        
        # Title
        qp.setPen(QPen(self.config.text_color))
        title_rect = QRectF(padding, padding, rect.width() - 2 * padding, title_height)
        qp.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self.config.title)
        
        # Category entries
        y = padding + title_height + 8
        x = padding
        
        for i, category in enumerate(self.config.categories):
            if not self._category_visible.get(category, True):
                continue
            
            # Color box (rounded with outline)
            box_rect = QRectF(x, y, box_size, box_size)
            color = self.config.category_colors.get(category, (0.5, 0.5, 0.5, 1.0))
            
            # Ensure color is a valid RGBA tuple
            if not isinstance(color, (tuple, list)) or len(color) < 3:
                color = (0.5, 0.5, 0.5, 1.0)
            elif len(color) == 3:
                color = (*color, 1.0)
            
            # Ensure values are in 0-1 range
            if any(c > 1.0 for c in color[:3]):
                color = tuple(c / 255.0 for c in color[:3]) + (color[3] if len(color) > 3 else 1.0,)
            
            # Debug: Log first few categories and their colors
            if i < 3:
                logger.debug(f"[LEGEND DRAW] Category '{category}' -> color tuple: {color}")
                try:
                    r, g, b = float(color[0]), float(color[1]), float(color[2])
                    a = float(color[3]) if len(color) > 3 else 1.0
                    qc_test = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
                    logger.debug(f"[LEGEND DRAW]   -> QColor: rgb=({qc_test.red()}, {qc_test.green()}, {qc_test.blue()}), alpha={qc_test.alpha()}, valid={qc_test.isValid()}")
                except Exception as e:
                    logger.warning(f"[LEGEND DRAW]   -> Failed to create QColor: {e}")
            
            # Create color with explicit RGBA values
            try:
                r, g, b = float(color[0]), float(color[1]), float(color[2])
                a = float(color[3]) if len(color) > 3 else 1.0
                # Use integer RGB values (0-255) for more reliable color creation
                swatch_color = QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255))
            except Exception as e:
                logger.warning(f"Failed to create color from {color}: {e}")
                swatch_color = QColor(128, 128, 128, 255)
            
            # Save painter state
            qp.save()
            
            # Set composition mode to SourceOver for proper color rendering
            qp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            
            # Create rounded rectangle path
            swatch_path = QPainterPath()
            swatch_radius = float(getattr(self.config, 'swatch_corner_radius', 5.0))
            swatch_path.addRoundedRect(box_rect, swatch_radius, swatch_radius)
            
            # Fill with color using brush
            qp.setBrush(QBrush(swatch_color))
            qp.setPen(Qt.PenStyle.NoPen)
            qp.drawPath(swatch_path)
            
            # Draw outline
            outline_color = getattr(self.config, 'bar_outline_color', self.config.border_color)
            outline_width = float(getattr(self.config, 'bar_outline_width', 2.0))
            qp.setPen(QPen(outline_color, outline_width))
            qp.setBrush(QBrush())  # No brush for outline
            qp.drawPath(swatch_path)
            
            # Restore painter state
            qp.restore()
            
            # Label - use alias if available, otherwise show code
            label_text = self.config.category_labels.get(category, str(category))
            label_x = x + box_size + 6
            label_y = y + fm.ascent()
            
            # Faded text if category is not visible
            if not self._category_visible.get(category, True):
                qp.setPen(QPen(QColor(150, 150, 150)))
            else:
                qp.setPen(QPen(self.config.text_color))
            
            qp.drawText(int(label_x), int(label_y), label_text)
            
            # Hover highlight
            if self._hover_category == category:
                highlight_rect = QRectF(x - 2, y - 2, rect.width() - x + 4, row_height + 4)
                qp.fillRect(highlight_rect, QColor(255, 255, 255, 30))
            
            y += row_height + row_spacing
        
        # Auto-resize height if vertical
        if self._orientation == "vertical":
            min_height = padding + title_height + 8 + len(self.config.categories) * (row_height + row_spacing) + padding
            if self.height() < min_height:
                self.setMinimumHeight(int(min_height))
    
    # ========================================================================
    # Helper Methods - Math and Formatting
    # ========================================================================
    
    def _normalize_value(self, value: float) -> float:
        """Normalize value to [0, 1] range: n = (v - vmin) / (vmax - vmin)."""
        if self.config.vmax == self.config.vmin:
            return 0.5
        
        if self.config.log_scale:
            # Log scale normalization
            if value <= 0:
                return 0.0
            log_min = np.log10(max(1e-10, abs(self.config.vmin)))
            log_max = np.log10(max(1e-10, abs(self.config.vmax)))
            log_val = np.log10(abs(value))
            if log_max == log_min:
                return 0.5
            n = (log_val - log_min) / (log_max - log_min)
            return np.clip(n, 0.0, 1.0)
        else:
            # Linear normalization
            n = (value - self.config.vmin) / (self.config.vmax - self.config.vmin)
            return np.clip(n, 0.0, 1.0)
    
    def _effective_brightness_factor(self) -> int:
        try:
            factor = int(getattr(self.config, "colormap_brightness", 135))
        except Exception:
            factor = 135
        return max(100, min(200, factor))

    def _brighten_color(self, color: QColor) -> QColor:
        factor = self._effective_brightness_factor()
        if factor == 100:
            return color
        bright = QColor(color)
        bright = bright.lighter(factor)
        return bright
    
    def _denormalize_value(self, n: float) -> float:
        """Convert normalized value [0, 1] back to actual value."""
        n = np.clip(n, 0.0, 1.0)
        
        if self.config.log_scale:
            log_min = np.log10(max(1e-10, abs(self.config.vmin)))
            log_max = np.log10(max(1e-10, abs(self.config.vmax)))
            log_val = log_min + n * (log_max - log_min)
            return 10 ** log_val
        else:
            return self.config.vmin + n * (self.config.vmax - self.config.vmin)
    
    def _compute_tick_values(self) -> List[float]:
        """Compute evenly-spaced tick values."""
        n_ticks = self.config.tick_count
        
        if self.config.log_scale:
            # Logarithmic ticks
            log_min = np.log10(max(1e-10, abs(self.config.vmin)))
            log_max = np.log10(max(1e-10, abs(self.config.vmax)))
            log_ticks = np.linspace(log_min, log_max, n_ticks)
            ticks = [10 ** log_tick for log_tick in log_ticks]
        else:
            # Linear ticks
            ticks = np.linspace(self.config.vmin, self.config.vmax, n_ticks).tolist()
        
        return ticks
    
    def _format_value(self, value: float) -> str:
        """Format value for display."""
        # Honor explicit decimals if provided
        decimals = self.config.label_decimals
        if decimals is None:
            # Auto-format based on magnitude
            abs_val = abs(value)
            if abs_val >= 1000:
                decimals = 1
            elif abs_val >= 1:
                decimals = 2
            elif abs_val >= 0.01:
                decimals = 3
            else:
                decimals = 4
        try:
            if self.config.label_thousands_sep:
                return f"{value:,.{decimals}f}"
            else:
                return f"{value:.{decimals}f}"
        except Exception:
            # Fallback safe formatting
            return str(value)
    
    def _cache_colormap(self):
        """Cache a fully opaque colormap for rendering."""
        try:
            if self._custom_color_samples is not None:
                rgba = np.array(self._custom_color_samples, dtype=float)
                logger.debug(f"Caching colormap from custom color samples (shape={rgba.shape})")
            elif self._custom_colormap is not None:
                rgba = np.asarray(self._custom_colormap(np.linspace(0.0, 1.0, 256)))
                logger.debug(f"Caching colormap from custom colormap object")
            else:
                # Use the colormap name from config
                cmap_name = getattr(self.config, 'cmap_name', 'viridis')
                logger.debug(f"Caching colormap from name: '{cmap_name}'")
                try:
                    # Try to get the colormap - handle both string names and registered colormaps
                    try:
                        cmap = cm.get_cmap(cmap_name, 256)
                    except ValueError:
                        # If colormap name not found, try with different case or common variations
                        cmap_name_lower = cmap_name.lower()
                        if cmap_name_lower == 'grey' or cmap_name_lower == 'gray':
                            cmap = cm.get_cmap('gray', 256)
                        else:
                            raise
                    rgba = np.asarray(cmap(np.linspace(0.0, 1.0, 256)))
                    logger.info(f"Successfully loaded colormap '{cmap_name}' -> '{getattr(cmap, 'name', 'unknown')}'")
                except Exception as e:
                    logger.error(f"Failed to load colormap '{cmap_name}', falling back to 'viridis': {e}", exc_info=True)
                    cmap = cm.get_cmap('viridis', 256)
                    rgba = np.asarray(cmap(np.linspace(0.0, 1.0, 256)))

            if rgba.ndim != 2 or rgba.shape[1] < 3:
                raise ValueError("Invalid colormap samples")

            if rgba.shape[1] == 3:
                rgba = np.column_stack([rgba, np.ones((rgba.shape[0], 1), dtype=float)])

            rgba = np.clip(rgba, 0.0, 1.0)
            rgba[:, 3] = 1.0  # force full opacity

            self._cached_cmap = mcolors.ListedColormap(rgba, name=f"{self.config.cmap_name}_cached")
        except Exception:
            fallback = np.asarray(cm.get_cmap("viridis", 256)(np.linspace(0.0, 1.0, 256)))
            fallback[:, 3] = 1.0
            self._cached_cmap = mcolors.ListedColormap(fallback, name="viridis_cached")
    
    def _get_bar_rect_continuous(self, rect: QRect) -> QRectF:
        """Get the color bar rectangle for continuous mode."""
        padding = self._padding
        font = QFont(self.config.font_family, self.config.font_size)
        fm = QFontMetrics(font)
        title_height = fm.height() + 4
        
        if self._orientation == "vertical":
            bar_top = padding + title_height + 8
            bar_left = padding
            bar_width = self._bar_width
            bar_height = rect.height() - bar_top - padding - 40
            return QRectF(bar_left, bar_top, bar_width, bar_height)
        else:
            bar_top = padding + title_height + 8
            bar_left = padding + 40
            bar_width = rect.width() - bar_left - padding - 40
            bar_height = self._bar_width
            return QRectF(bar_left, bar_top, bar_width, bar_height)
    
    def _value_at_position(self, pos: QPoint) -> Optional[float]:
        """Get value at mouse position (continuous mode)."""
        if self.config.type != LegendType.CONTINUOUS:
            return None
        
        rect = self.rect()
        bar_rect = self._get_bar_rect_continuous(rect)
        
        if self._orientation == "vertical":
            if bar_rect.left() <= pos.x() <= bar_rect.right():
                if bar_rect.top() <= pos.y() <= bar_rect.bottom():
                    # Convert y to normalized value
                    n = 1.0 - (pos.y() - bar_rect.top()) / bar_rect.height()
                    if self.config.reverse:
                        n = 1.0 - n
                    return self._denormalize_value(n)
        else:
            if bar_rect.top() <= pos.y() <= bar_rect.bottom():
                if bar_rect.left() <= pos.x() <= bar_rect.right():
                    # Convert x to normalized value
                    n = (pos.x() - bar_rect.left()) / bar_rect.width()
                    if self.config.reverse:
                        n = 1.0 - n
                    return self._denormalize_value(n)
        
        return None
    
    def _category_at_position(self, pos: QPoint) -> Optional[Union[str, int, float]]:
        """Get category at mouse position (discrete mode)."""
        if self.config.type != LegendType.DISCRETE:
            return None
        
        font = QFont(self.config.font_family, self.config.font_size)
        fm = QFontMetrics(font)
        padding = self._padding
        title_height = fm.height() + 4
        box_size = self._category_box_size
        row_height = max(box_size + 4, fm.height() + 4)
        row_spacing = 2
        
        y_start = padding + title_height + 8
        
        for i, category in enumerate(self.config.categories):
            y = y_start + i * (row_height + row_spacing)
            if y <= pos.y() <= y + row_height:
                if padding <= pos.x() <= self.width() - padding:
                    return category
        
        return None
    
    def _draw_resize_handle(self, qp: QPainter, rect: QRect):
        """Draw resize handle in bottom-right corner."""
        handle_size = self._resize_handle_size
        handle_rect = QRectF(
            rect.right() - handle_size,
            rect.bottom() - handle_size,
            handle_size,
            handle_size
        )
        
        # Draw diagonal lines
        qp.setPen(QPen(self.config.border_color, 1.5))
        margin = 3
        qp.drawLine(
            int(handle_rect.left() + margin),
            int(handle_rect.bottom() - margin),
            int(handle_rect.right() - margin),
            int(handle_rect.top() + margin)
        )
    
    def _in_resize_handle(self, pos: QPoint) -> bool:
        """Check if position is in resize handle."""
        rect = self.rect()
        handle_size = self._resize_handle_size
        handle_rect = QRect(
            rect.right() - handle_size,
            rect.bottom() - handle_size,
            handle_size,
            handle_size
        )
        return handle_rect.contains(pos)

    def _get_button_at_pos(self, pos: QPoint) -> Optional[str]:
        """Get the button key at a given position (multi-mode)."""
        for key, rect in self._button_rects.items():
            if rect.contains(pos):
                return key
        return None

    def _handle_button_click(self, button_key: str) -> None:
        """Handle a button click in multi-mode."""
        if button_key == 'add':
            # Emit signal to request add dialog
            self.add_element_requested.emit()
            logger.debug("Add element button clicked")
        elif button_key.startswith('vis_'):
            # Toggle element visibility
            element_id = button_key[4:]  # Remove 'vis_' prefix
            elem = self.get_element(element_id)
            if elem:
                elem.visible = not elem.visible
                self.element_visibility_changed.emit(element_id, elem.visible)
                logger.debug("Toggle visibility: %s -> %s", element_id, elem.visible)
                self.update()
        elif button_key.startswith('rem_'):
            # Remove element
            element_id = button_key[4:]  # Remove 'rem_' prefix
            self.remove_element(element_id)

    # ========================================================================
    # Mouse Interaction
    # ========================================================================
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()

            # Check multi-mode button clicks first
            if self._multi_mode:
                clicked_button = self._get_button_at_pos(pos)
                if clicked_button:
                    self._handle_button_click(clicked_button)
                    event.accept()
                    return

            if self._in_resize_handle(pos):
                self._resizing = True
                self._resize_start_size = self.size()
                self._resize_start_pos = event.globalPosition().toPoint()
                logger.debug("Legend resize initiated at %s", pos)
            else:
                self._dragging = True
                self._drag_start_pos = event.globalPosition().toPoint() - self.pos()
                logger.debug("Legend drag initiated at %s", pos)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            event.accept()
            try:
                logger.debug("Legend context menu requested via right-click")
                self._exec_context_menu(event.globalPosition().toPoint())
            except Exception:
                pass
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events."""
        pos = event.position().toPoint()

        if self._resizing:
            if self._resize_start_size and self._resize_start_pos:
                delta = event.globalPosition().toPoint() - self._resize_start_pos
                new_width = max(self.minimumWidth(), self._resize_start_size.width() + delta.x())
                new_height = max(self.minimumHeight(), self._resize_start_size.height() + delta.y())
                self.resize(new_width, new_height)
                try:
                    self.size_changed.emit(self.width(), self.height())
                except Exception:
                    pass
            event.accept()
            return

        if self._dragging:
            if self._drag_start_pos:
                new_pos = event.globalPosition().toPoint() - self._drag_start_pos
                parent = self.parentWidget()
                if parent is not None:
                    max_x = max(0, parent.width() - self.width())
                    max_y = max(0, parent.height() - self.height())
                    new_pos.setX(max(0, min(max_x, new_pos.x())))
                    new_pos.setY(max(0, min(max_y, new_pos.y())))
                self.move(new_pos)
                try:
                    self.floating_position_changed.emit(self.x(), self.y())
                except Exception:
                    pass
            event.accept()
            return

        # Update hover state
        self._hover_position = pos

        # Multi-mode hover handling
        if self._multi_mode:
            old_hover = self._hover_button
            self._hover_button = self._get_button_at_pos(pos)
            if self._hover_button != old_hover:
                self.update()
            # Update cursor for buttons
            if self._hover_button:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            elif self._in_resize_handle(pos):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            return

        if self.config.type == LegendType.CONTINUOUS:
            old_hover = self._hover_value
            self._hover_value = self._value_at_position(pos)
            if self._hover_value != old_hover:
                self.update()
                if self._hover_value is not None:
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"{self.config.title}: {self._hover_value:.4f}",
                        self
                    )
        elif self.config.type == LegendType.DISCRETE:
            old_category = self._hover_category
            self._hover_category = self._category_at_position(pos)
            if self._hover_category != old_category:
                self.update()
                if self._hover_category is not None:
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"{self._hover_category}",
                        self
                    )

        # Update cursor
        if self._in_resize_handle(pos):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.LeftButton:
            was_dragging = self._dragging
            was_resizing = self._resizing
            self._dragging = False
            self._resizing = False

            if was_resizing:
                try:
                    self.size_changed.emit(self.width(), self.height())
                except Exception:
                    pass
                logger.debug("Legend resize completed new_size=%sx%s", self.width(), self.height())

            if was_dragging:
                anchor = self._suggest_anchor()
                if anchor != "floating":
                    self._dock_anchor = anchor
                    try:
                        self.dock_requested.emit(anchor)
                    except Exception:
                        pass
                    logger.debug("Legend drag completed snapped to anchor=%s", anchor)
                else:
                    try:
                        self.floating_position_changed.emit(self.x(), self.y())
                    except Exception:
                        pass
                    logger.debug("Legend drag completed floating_position=(%s,%s)", self.x(), self.y())

            # Handle click interactions when it was a simple click
            if not was_dragging and not was_resizing:
                pos = event.position().toPoint()

                if self.config.type == LegendType.DISCRETE:
                    category = self._category_at_position(pos)
                    if category is not None:
                        self.toggle_category_visibility(category)
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click for category color editing (discrete mode)."""
        if event.button() == Qt.MouseButton.LeftButton and self.config.type == LegendType.DISCRETE:
            pos = event.position().toPoint()
            category = self._category_at_position(pos)
            
            if category is not None:
                logger.debug("Legend category colour edit requested for %s", category)
                # Get current color
                current_rgba = self.config.category_colors.get(category, (0.5, 0.5, 0.5, 1.0))
                current_color = QColor.fromRgbF(*current_rgba)
                
                # Open color dialog
                new_color = QColorDialog.getColor(current_color, self, f"Color for {category}")
                
                if new_color.isValid():
                    rgba = new_color.getRgbF()
                    new_rgba_tuple = (rgba[0], rgba[1], rgba[2], rgba[3])
                    self.config.category_colors[category] = new_rgba_tuple
                    self.update()
                    logger.debug(
                        "Legend category %s colour updated to rgba(%s, %s, %s, %s)",
                        category,
                        new_color.red(),
                        new_color.green(),
                        new_color.blue(),
                        new_color.alpha(),
                    )
                    # Emit signal to notify renderer/meshes of color change
                    try:
                        self.category_color_changed.emit(category, new_rgba_tuple)
                        logger.info(f"Emitted category_color_changed signal for '{category}' -> {new_rgba_tuple}")
                    except Exception as e:
                        logger.warning(f"Failed to emit category_color_changed: {e}")

    def _suggest_anchor(self) -> str:
        """Return anchor identifier if close enough to a corner, else 'floating'."""
        parent = self.parentWidget()
        if parent is None:
            return "floating"

        parent_w = parent.width()
        parent_h = parent.height()
        if parent_w <= 0 or parent_h <= 0:
            return "floating"

        candidate_positions = {
            "top_left": (0, 0),
            "top_right": (parent_w - self.width(), 0),
            "bottom_left": (0, parent_h - self.height()),
            "bottom_right": (parent_w - self.width(), parent_h - self.height()),
        }

        current_x = self.x()
        current_y = self.y()

        best_anchor = "floating"
        best_distance = float("inf")

        for anchor, (target_x, target_y) in candidate_positions.items():
            dx = current_x - target_x
            dy = current_y - target_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_distance:
                best_distance = dist
                best_anchor = anchor

        if best_distance <= self._snap_threshold:
            return best_anchor

        return "floating"
    
    # ========================================================================
    # Context Menu
    # ========================================================================
    
    def _show_context_menu(self, event: QMouseEvent):
        """Legacy entry point for right-click; delegate to unified menu builder."""
        try:
            self._exec_context_menu(event.globalPosition().toPoint())
        except Exception:
            pass
    
    # ========================================================================
    # Export
    # ========================================================================
    
    def export_png(self, path: Optional[str] = None, dpi: int = 300):
        """Export legend to PNG at specified DPI."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Legend to PNG",
                "legend.png",
                "PNG Images (*.png)"
            )
            if not path:
                return
        
        # Render to high-resolution pixmap
        scale = dpi / 96.0  # Qt default DPI
        img_size = QSize(int(self.width() * scale), int(self.height() * scale))
        pixmap = QPixmap(img_size)
        pixmap.fill(QColor(255, 255, 255, 0))  # Transparent background
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.scale(scale, scale)
        
        # Render the widget using render() method (Qt's built-in rendering)
        self.render(painter, QPoint(0, 0))
        
        painter.end()
        
        # Save
        pixmap.save(path, "PNG", 100)
        
        QMessageBox.information(self, "Export Complete", f"Legend exported to:\n{path}")

    def sync_with_renderer_lut(self, renderer):
        """
        Rebuilds the legend gradient directly from the renderer's active LUT.
        Ensures colours match the 3D mesh exactly (no gamma or normalization drift).
        """
        try:
            lut = getattr(renderer, "active_scalar_lut", None)
            if lut is None:
                actors = getattr(renderer.plotter, "actors", {})
                if actors:
                    first_actor = list(actors.values())[0]
                    mapper = getattr(first_actor, "mapper", None)
                    if mapper is None and hasattr(first_actor, "GetMapper"):
                        mapper = first_actor.GetMapper()
                    if mapper is not None:
                        try:
                            lut = mapper.GetLookupTable()
                        except Exception:
                            try:
                                lut = mapper.lookup_table
                            except Exception:
                                lut = None
            if lut is None:
                logger.debug("No LUT found for legend sync.")
                return

            n = lut.GetNumberOfTableValues()
            import numpy as np

            lut_rgba = np.ones((max(1, n), 4), dtype=float)
            try:
                table = lut.GetTable()
            except Exception:
                table = None

            if table is not None:
                try:
                    lut_rgba = np.array([table.GetTuple4(i) for i in range(n)], dtype=float) / 255.0
                except Exception:
                    lut_rgba = np.ones((max(1, n), 4), dtype=float)

            if table is None or lut_rgba.shape[0] != n:
                values = []
                for i in range(n):
                    try:
                        rgba = lut.GetTableValue(i)
                    except Exception:
                        try:
                            rgba = lut.GetColor(float(i) / max(1, n - 1)) + (lut.GetOpacity(float(i) / max(1, n - 1)),)
                        except Exception:
                            rgba = (1.0, 1.0, 1.0, 1.0)
                    values.append(rgba)
                lut_rgba = np.array(values, dtype=float)
                if lut_rgba.shape[1] == 3:
                    lut_rgba = np.column_stack([lut_rgba, np.ones((lut_rgba.shape[0], 1), dtype=float)])
            lut_rgba = np.clip(lut_rgba, 0.0, 1.0)

            def lut_cmap(norm_val: float):
                idx = int(norm_val * (n - 1))
                idx = max(0, min(n - 1, idx))
                return lut_rgba[idx]

            self._cached_cmap = lut_cmap
            self.update()
            logger.debug(f"Legend synced with renderer LUT ({n} entries).")
        except Exception as e:
            logger.debug(f"Legend LUT sync failed: {e}")
