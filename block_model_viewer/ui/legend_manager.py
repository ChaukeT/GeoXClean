"""
Controller layer for synchronising renderer state with the modern legend widget.

The LegendManager listens to renderer updates, prepares legend payloads and drives
visual refinements (theme, transitions, caching) so the renderer only pushes state
changes and never paints directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import logging
import weakref

import numpy as np
from PyQt6.QtCore import QObject, QPropertyAnimation, QEasingCurve, QEvent, pyqtSignal
from PyQt6.QtGui import QColor
from matplotlib import cm
import matplotlib.colors as mcolors

from .legend_widget import LegendWidget, LegendType
from .legend_types import LegendElement, LegendElementType, MultiLegendConfig


if TYPE_CHECKING:
    from .multi_legend_widget import MultiLegendWidget  # pragma: no cover
    from ..visualization.renderer import Renderer  # pragma: no cover


logger = logging.getLogger(__name__)


@dataclass
class LegendPayload:
    """Lightweight structure describing the legend visual state."""

    layer: Optional[str]
    property: Optional[str]
    title: str
    mode: str
    colormap: str
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    log_scale: bool = False
    reverse: bool = False
    data: Optional[np.ndarray] = None
    categories: List[Union[str, int, float]] = field(default_factory=list)
    category_colors: Dict[Union[str, int, float], Tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    subtitle: str = ""


@dataclass
class LegendState:
    """Persisted legend overlay configuration."""

    anchor: str = "top_right"  # 'floating', 'top_left', etc.
    position: Tuple[int, int] = (0, 0)  # Only used when anchor == 'floating'
    size: Tuple[int, int] = (300, 200)
    margin: int = 24
    visible: bool = True
    orientation: str = "vertical"
    background_rgba: Tuple[int, int, int, int] = (15, 15, 20, 180)


class LegendManager(QObject):
    """
    Synchronises the renderer (model layer) with the painter-driven LegendWidget (view layer).

    Responsibilities:
        • Listen for renderer property/colormap churn and prepare legend payloads.
        • Manage adaptive theming against scene background.
        • Animate legend updates for a polished experience.
        • Cache context (layer/property) for interaction callbacks.
    """

    # Signal for legend state changes (used by sync hub)
    legend_changed = pyqtSignal(dict)
    visibility_changed = pyqtSignal(bool)  # Emitted when legend visibility changes

    def __init__(self, renderer: "Renderer", legend_widget: Optional[LegendWidget] = None):
        super().__init__()
        self.renderer = renderer
        self.widget = legend_widget

        self._animation: Optional[QPropertyAnimation] = None
        self._pending_payloads: List[LegendPayload] = []
        self._current_payload: Optional[LegendPayload] = None
        self._state = LegendState()
        self._visible: bool = True
        self._custom_background: Optional[QColor] = None
        self._resize_filter: Optional["_LegendResizeFilter"] = None
        self._margin_px: int = self._state.margin
        
        # LAG FIX: Add enabled flag to skip updates when disabled
        self._enabled: bool = True

        try:
            initial_state = getattr(renderer, "legend_state", None)
            if isinstance(initial_state, dict):
                self._load_state(initial_state)
        except Exception:
            pass

        # Cached statistics for recoloring (colormap swap without recomputing)
        self._current_data: Optional[np.ndarray] = None
        self._current_property: Optional[str] = None
        self._categories: Optional[List] = None
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None
        self._colormap: str = "viridis"

        # Discrete detection thresholds (configurable)
        self.discrete_unique_max: int = 20
        self.discrete_fraction_max: float = 0.10
        
        # Current label namespace for category label aliasing
        self._current_namespace: Optional[str] = None
        self._registry_ref: Optional[Any] = None

        if legend_widget is not None:
            self.bind_widget(legend_widget)

    # ---------------------------------------------------------------------#
    # Wiring
    # ---------------------------------------------------------------------#
    def attach_registry(self, registry: Any) -> None:
        """
        Attach DataRegistry for category label aliasing.
        
        Args:
            registry: DataRegistry instance
        """
        self._registry_ref = registry
        
        # Connect to registry signal for label map changes
        try:
            if hasattr(registry, 'categoryLabelMapsChanged'):
                registry.categoryLabelMapsChanged.connect(self._on_label_maps_changed)
                logger.debug("Connected to DataRegistry.categoryLabelMapsChanged")
        except Exception as e:
            logger.debug(f"Could not connect to registry label map signal: {e}")
    
    def bind_widget(self, widget: LegendWidget) -> None:
        """Attach the view layer after it is created."""
        self.widget = widget
        self._visible = bool(self._state.visible)
        self._setup_animation()
        self._apply_theme_from_renderer()
        try:
            orientation = self._state.orientation or getattr(self.renderer, "legend_position", "vertical")
            self.widget.set_orientation(orientation)
        except Exception:
            pass

        try:
            self.widget.background_color_changed.connect(self._on_background_color_changed)
            self.widget.dock_requested.connect(self._on_dock_requested)
            self.widget.floating_position_changed.connect(self._on_floating_position_changed)
            self.widget.size_changed.connect(self._on_widget_resized)
            # Category label editing
            self.widget.category_label_changed.connect(self._on_category_label_changed)
            # Category color editing - propagate to renderer meshes
            try:
                self.widget.category_color_changed.connect(self._on_category_color_changed)
            except Exception:
                pass
            # When the user toggles the legend orientation via the widget's UI,
            # update the manager's state and apply the orientation. Without this
            # connection the legend would revert to its previous orientation on
            # the next update because LegendManager doesn't know about the
            # change.
            try:
                self.widget.orientation_changed.connect(self._on_widget_orientation_changed)
            except Exception:
                pass
        except Exception:
            pass

        try:
            self.widget.set_current_anchor(self._state.anchor)
            self.widget.apply_layout(
                anchor=self._state.anchor,
                position=self._state.position,
                margin=self._state.margin,
                size=self._state.size,
            )
        except Exception:
            pass

        parent = self.widget.parentWidget()
        if parent is not None and self._resize_filter is None:
            try:
                self._resize_filter = _LegendResizeFilter(self)
                parent.installEventFilter(self._resize_filter)
            except Exception:
                self._resize_filter = None

        if self._pending_payloads:
            for payload in self._pending_payloads:
                self._apply_payload(payload, animate=False)
            self._pending_payloads.clear()

        if not self._visible:
            self.widget.hide()
        else:
            # Ensure the legend is visible and at a sensible opacity when bound.
            # Older state could have left the widget at 0.0 opacity (invisible).
            try:
                # Prefer the saved/desired opacity if available via renderer state,
                # otherwise default to fully opaque for clarity.
                desired_opacity = getattr(self.renderer, "legend_background_opacity", None)
                if desired_opacity is None:
                    desired_opacity = 1.0
                # Clamp to sensible range
                desired_opacity = max(0.05, min(1.0, float(desired_opacity)))
                self.widget.setWindowOpacity(desired_opacity)
            except Exception:
                try:
                    self.widget.setWindowOpacity(1.0)
                except Exception:
                    pass
            try:
                if not self.widget.isVisible():
                    self.widget.show()
                self._ensure_position()
            except Exception:
                pass

    def _setup_animation(self) -> None:
        if self.widget is None:
            return
        self._animation = QPropertyAnimation(self.widget, b"windowOpacity", self.widget)
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

    # ---------------------------------------------------------------------#
    # Public API
    # ---------------------------------------------------------------------#
    def set_discrete_thresholds(self, unique_max: Optional[int] = None, fraction_max: Optional[float] = None):
        if unique_max is not None and unique_max > 0:
            self.discrete_unique_max = int(unique_max)
        if fraction_max is not None and 0 < fraction_max <= 1:
            self.discrete_fraction_max = float(fraction_max)

    def update_from_renderer(self, *, force: bool = False, reason: str = "") -> None:
        """Pull the current renderer context and drive the legend widget."""
        if self.renderer is not None:
            self._visible = bool(getattr(self.renderer, "legend_visible", self._visible))
        try:
            payload = self._gather_renderer_state()
        except Exception as exc:
            try:
                exc_msg = str(exc)
                logger.debug("Failed to build legend payload: %s", exc_msg)
            except Exception:
                logger.debug("Failed to build legend payload: <unprintable error>")
            return

        if payload is None:
            # Don't overwrite existing valid legend data with "No data"
            # This can happen when drillholes are active but _gather_renderer_state returns None
            # because drillhole legend is managed separately via update_discrete/update_continuous
            if self.widget is not None:
                # Check if widget already has valid data
                has_valid_data = (
                    (self.widget.config.type == LegendType.DISCRETE and self.widget.config.categories) or
                    (self.widget.config.type == LegendType.CONTINUOUS and
                     self.widget.config.vmin is not None and self.widget.config.vmax is not None)
                )
                if not has_valid_data:
                    self.widget.set_empty_state("No data")
                else:
                    logger.debug("Preserving existing legend data (payload was None but widget has valid data)")
            else:
                self._pending_payloads.clear()
            return

        if self.widget is None:
            self._pending_payloads.append(payload)
            return

        if not force and self._current_payload == payload:
            return

        self._apply_payload(payload, animate=True)
        self._current_payload = payload

    def detect_and_update(self, property_name: str, data: np.ndarray, cmap_name: str = "viridis"):
        """Backwards-compatible entry point used by legacy call sites."""
        if data is None or len(data) == 0:
            return

        arr = np.asarray(data)
        if np.issubdtype(arr.dtype, np.number):
            finite_data = arr[np.isfinite(arr)]
            if len(finite_data) == 0:
                return
            unique_vals = np.unique(finite_data)
            if (
                len(unique_vals) <= self.discrete_unique_max
                and len(unique_vals) <= int(len(finite_data) * self.discrete_fraction_max)
            ):
                self.update_discrete(property_name, unique_vals.tolist(), cmap_name=cmap_name)
            else:
                self.update_continuous(property_name, arr, cmap_name)
        else:
            unique_vals = np.unique(np.array(arr, dtype=str))
            unique_vals = unique_vals[(unique_vals != "") & (unique_vals != "nan")]
            self.update_discrete(property_name, unique_vals.tolist(), cmap_name=cmap_name)

    def update_continuous(
        self,
        property_name: str,
        data: np.ndarray,
        cmap_name: Union[str, Any] = "viridis",  # Can be string or colormap object
        log_scale: bool = False,
        subtitle: str = "",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center_zero: bool = False,
    ):
        """
        Update continuous legend with support for divergent maps.
        
        Args:
            property_name: Name of the property
            data: Property values array
            cmap_name: Colormap name or colormap object
            log_scale: Whether to use logarithmic scale
            subtitle: Subtitle text
            vmin: Minimum value (if None, computed from data)
            vmax: Maximum value (if None, computed from data)
            center_zero: If True, display as divergent map (zero-centered)
        """
        # LAG FIX: Skip update if legend manager is disabled
        if not self._enabled:
            return
        if data is None or len(data) == 0:
            return

        array_data = np.asarray(data)

        if self.widget is None:
            payload = LegendPayload(
                layer=None,
                property=property_name,
                title=property_name,
                mode="continuous",
                colormap=cmap_name,
                data=array_data,
                log_scale=log_scale,
                subtitle=subtitle,
                vmin=vmin,
                vmax=vmax,
            )
            self._pending_payloads.append(payload)
            self._current_data = array_data
            self._current_property = property_name
            return

        finite_data = array_data[np.isfinite(array_data)]
        if len(finite_data) == 0:
            return

        # Calculate vmin/vmax if not provided
        if vmin is None:
            vmin = float(np.nanmin(finite_data))
        if vmax is None:
            vmax = float(np.nanmax(finite_data))
        
        # For divergent maps, ensure symmetric range around zero
        if center_zero:
            max_abs = max(abs(vmin), abs(vmax))
            vmin = -max_abs
            vmax = max_abs
        
        if vmin >= vmax:
            vmax = vmin + 1.0

        # Handle colormap object vs string
        if hasattr(cmap_name, 'colors') or hasattr(cmap_name, '__call__'):
            # It's a colormap object - extract colors for custom colormap
            try:
                # Sample colors from the colormap
                samples = np.linspace(0.0, 1.0, 256)
                if hasattr(cmap_name, '__call__'):
                    rgba_samples = np.array([cmap_name(s) for s in samples])
                else:
                    rgba_samples = cmap_name(samples)
                # Extract colormap name - try to get a meaningful name
                cmap_str_name = 'custom'
                if hasattr(cmap_name, 'name'):
                    cmap_str_name = cmap_name.name
                    # Remove '_custom' suffix if present to match property panel names
                    if cmap_str_name.endswith('_custom'):
                        cmap_str_name = cmap_str_name[:-7]
                elif isinstance(cmap_name, str):
                    cmap_str_name = cmap_name
                
                # Pass as color_samples to widget
                self.widget.set_continuous(
                    title=property_name,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=cmap_str_name,
                    tick_count=max(3, getattr(self.renderer, "legend_label_count", 6)),
                    log_scale=log_scale,
                    reverse=getattr(self.widget.config, "reverse", False),
                    data=array_data,
                    color_samples=rgba_samples,
                )
            except Exception as e:
                logger.warning(f"Failed to extract colors from colormap object, using string name: {e}")
                # Extract colormap name properly
                cmap_str = 'viridis'
                if hasattr(cmap_name, 'name'):
                    cmap_str = cmap_name.name
                    if cmap_str.endswith('_custom'):
                        cmap_str = cmap_str[:-7]
                elif isinstance(cmap_name, str):
                    cmap_str = cmap_name
                else:
                    cmap_str = str(cmap_name)
                
                self.widget.set_continuous(
                    title=property_name,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=cmap_str,
                    tick_count=max(3, getattr(self.renderer, "legend_label_count", 6)),
                    log_scale=log_scale,
                    reverse=getattr(self.widget.config, "reverse", False),
                    data=array_data,
                )
        else:
            # It's a string colormap name
            logger.info(f"LegendManager.update_continuous: property={property_name}, cmap_name='{cmap_name}', range=({vmin:.2f}, {vmax:.2f})")
            try:
                self.widget.set_continuous(
                    title=property_name,
                    vmin=vmin,
                    vmax=vmax,
                    cmap_name=cmap_name,
                    tick_count=max(3, getattr(self.renderer, "legend_label_count", 6)),
                    log_scale=log_scale,
                    reverse=getattr(self.widget.config, "reverse", False),
                    data=array_data,
                )
                logger.info(f"LegendManager.update_continuous: Successfully called widget.set_continuous")
            except Exception as e:
                logger.error(f"LegendManager.update_continuous: Error calling widget.set_continuous: {e}", exc_info=True)
        
        # Ensure colormap is cached - set_continuous calls _cache_colormap, but verify it worked
        if not hasattr(self.widget, "_cached_cmap") or self.widget._cached_cmap is None:
            try:
                logger.warning(f"Colormap not cached after set_continuous, caching now: {cmap_name}")
                self.widget._cache_colormap()
            except Exception as e:
                logger.error(f"Failed to cache colormap: {e}", exc_info=True)
        
        self.widget.set_subtitle(subtitle)
        try:
            if hasattr(self.widget, "sync_with_renderer_lut") and self.renderer is not None:
                self.widget.sync_with_renderer_lut(self.renderer)
        except Exception as exc:
            try:
                exc_msg = str(exc)
                logger.debug("LegendManager LUT sync attempt failed: %s", exc_msg)
            except Exception:
                logger.debug("LegendManager LUT sync attempt failed: <unprintable error>")
        
        # --- Ensure visibility and refresh after updating continuous legend ---
        try:
            if not self.widget.isVisible():
                self.widget.show()
                self.widget.raise_()
            
            # Verify the colormap was set correctly
            widget_cmap_name = getattr(self.widget.config, 'cmap_name', 'unknown')
            logger.info(f"LegendManager: Updated continuous legend: property='{property_name}', range=({vmin:.2f}, {vmax:.2f}), requested_cmap='{cmap_name}', widget_cmap='{widget_cmap_name}'")
            
            # Force repaint to ensure color bar is drawn with correct colormap
            # Use update() to schedule paint event (avoid repaint() to prevent recursive repaints)
            self.widget.update()
            
            # Also process events to ensure the paint happens
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
            
            if widget_cmap_name != cmap_name and isinstance(cmap_name, str):
                logger.warning(f"Colormap mismatch! Requested '{cmap_name}' but widget has '{widget_cmap_name}'")
        except Exception as e:
            logger.error(f"Failed to show/update legend widget: {e}", exc_info=True)

        self._current_data = array_data
        self._current_property = property_name

        # Store current payload for state capture (layout exports, etc.)
        # Extract the final colormap name string
        final_cmap_name = cmap_name
        if hasattr(cmap_name, 'name'):
            final_cmap_name = cmap_name.name
            if isinstance(final_cmap_name, str) and final_cmap_name.endswith('_custom'):
                final_cmap_name = final_cmap_name[:-7]
        elif not isinstance(cmap_name, str):
            final_cmap_name = str(cmap_name)

        self._current_payload = LegendPayload(
            layer=None,
            property=property_name,
            title=property_name,
            mode="continuous",
            colormap=final_cmap_name if isinstance(final_cmap_name, str) else 'viridis',
            data=array_data,
            log_scale=log_scale,
            subtitle=subtitle,
            vmin=vmin,
            vmax=vmax,
        )

        # Sync multi-mode elements if active
        self._sync_multi_mode_continuous(property_name, vmin, vmax, cmap_name if isinstance(cmap_name, str) else 'viridis')

    def update_discrete(
        self,
        property_name: str,
        categories: List[Union[str, int, float]],
        category_colors: Optional[Dict[Union[str, int, float], Tuple[float, float, float, float]]] = None,
        cmap_name: Optional[Union[str, Any]] = None,  # Can be string or colormap object
        subtitle: str = "",
    ):
        # DEBUG: Log received parameters
        logger.debug(f"[LEGEND DEBUG] LegendManager.update_discrete called:")
        logger.debug(f"[LEGEND DEBUG]   property_name: {property_name}")
        logger.debug(f"[LEGEND DEBUG]   categories: {categories[:3] if len(categories) > 3 else categories}")
        logger.debug(f"[LEGEND DEBUG]   cmap_name: {cmap_name}")
        if category_colors:
            category_colors = self._normalize_category_colors(category_colors)
            logger.debug(f"[LEGEND DEBUG]   category_colors sample:")
            for cat, color in list(category_colors.items())[:3]:
                logger.debug(f"[LEGEND DEBUG]     {cat}: {color} (type: {type(color)})")
        
        # Determine namespace for category label aliasing
        self._current_namespace = self._determine_label_namespace(property_name)
        
        # LAG FIX: Skip update if legend manager is disabled
        if not self._enabled:
            return
        if not categories:
            return

        if self.widget is None:
            payload = LegendPayload(
                layer=None,
                property=property_name,
                title=property_name,
                mode="discrete",
                colormap=cmap_name or "tab20",
                categories=list(categories),
                category_colors=category_colors or {},
                subtitle=subtitle,
            )
            self._pending_payloads.append(payload)
            self._current_property = property_name
            self._current_data = None
            return

        # Handle colormap object vs string for discrete mode
        if cmap_name is not None and (hasattr(cmap_name, 'colors') or hasattr(cmap_name, '__call__')):
            # It's a colormap object - extract colors for categories
            try:
                # If category_colors not provided, generate from colormap
                if category_colors is None:
                    n = len(categories)
                    if n <= 1:
                        samples = [0.0]
                    else:
                        samples = np.linspace(0.0, 1.0, n)
                    category_colors = {}
                    for idx, cat in enumerate(categories):
                        if hasattr(cmap_name, '__call__'):
                            rgba = cmap_name(samples[idx])
                        else:
                            rgba = cmap_name(samples[idx])
                        category_colors[cat] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
                
                # DEBUG: Log before calling widget.set_discrete (colormap object path)
                logger.debug(f"[LEGEND DEBUG] LegendManager calling widget.set_discrete (colormap object path)")
                logger.debug(f"[LEGEND DEBUG]   Passing category_colors sample:")
                for cat, color in list(category_colors.items())[:3]:
                    logger.debug(f"[LEGEND DEBUG]     {cat}: {color}")
                
                self.widget.set_discrete(
                    title=property_name,
                    categories=categories,
                    category_colors=category_colors,
                    auto_colors=False,  # We're providing colors
                    cmap_name=getattr(cmap_name, 'name', 'custom'),
                )
            except Exception as e:
                logger.warning(f"Failed to extract colors from colormap object, using string name: {e}")
                cmap_str = getattr(cmap_name, 'name', 'tab20') if hasattr(cmap_name, 'name') else str(cmap_name)
                self.widget.set_discrete(
                    title=property_name,
                    categories=categories,
                    category_colors=category_colors,
                    auto_colors=(category_colors is None),
                    cmap_name=cmap_str,
                )
        else:
            # It's a string colormap name or None
            # DEBUG: Log before calling widget.set_discrete (string path)
            logger.debug(f"[LEGEND DEBUG] LegendManager calling widget.set_discrete (string colormap path)")
            logger.debug(f"[LEGEND DEBUG]   cmap_name: {cmap_name}")
            logger.debug(f"[LEGEND DEBUG]   auto_colors: {category_colors is None}")
            if category_colors:
                logger.debug(f"[LEGEND DEBUG]   Passing category_colors sample:")
                for cat, color in list(category_colors.items())[:3]:
                    logger.debug(f"[LEGEND DEBUG]     {cat}: {color}")
            
            self.widget.set_discrete(
                title=property_name,
                categories=categories,
                category_colors=category_colors,
                auto_colors=(category_colors is None),
                cmap_name=cmap_name,
            )
        
        # Apply category labels from registry
        self._apply_category_labels_from_registry()
        
        self.widget.set_subtitle(subtitle)
        self.widget.update()

        self._current_property = property_name
        self._current_data = None

        # Store current payload for state capture (layout exports, etc.)
        # Extract the final colormap name string
        final_cmap_name = cmap_name
        if hasattr(cmap_name, 'name'):
            final_cmap_name = cmap_name.name
        elif cmap_name is None:
            final_cmap_name = "tab20"
        elif not isinstance(cmap_name, str):
            final_cmap_name = str(cmap_name)

        self._current_payload = LegendPayload(
            layer=None,
            property=property_name,
            title=property_name,
            mode="discrete",
            colormap=final_cmap_name if isinstance(final_cmap_name, str) else 'tab20',
            categories=list(categories),
            category_colors=category_colors or {},
            subtitle=subtitle,
        )

        # Sync multi-mode elements if active
        self._sync_multi_mode_discrete(property_name, categories, category_colors or {})

    def _sync_multi_mode_continuous(
        self,
        property_name: str,
        vmin: float,
        vmax: float,
        cmap_name: str,
    ) -> None:
        """Sync continuous data to matching multi-mode elements."""
        if self.widget is None or not self.widget.is_multi_mode():
            return

        # Find elements that match this property
        for element in self.widget.get_elements():
            if element.element_type != LegendElementType.CONTINUOUS:
                continue

            # Match by property name in title or source_property
            if (property_name in element.title or
                element.source_property == property_name or
                property_name.lower() in element.id.lower()):

                # Update the element
                element.vmin = vmin
                element.vmax = vmax
                element.cmap_name = cmap_name
                logger.info(f"Synced multi-mode element '{element.id}': vmin={vmin}, vmax={vmax}, cmap={cmap_name}")

        self.widget.update()

    def _sync_multi_mode_discrete(
        self,
        property_name: str,
        categories: List,
        category_colors: Dict,
    ) -> None:
        """Sync discrete data to matching multi-mode elements."""
        if self.widget is None or not self.widget.is_multi_mode():
            return

        # Find elements that match this property
        for element in self.widget.get_elements():
            if element.element_type != LegendElementType.DISCRETE:
                continue

            # Match by property name in title or source_property
            if (property_name in element.title or
                element.source_property == property_name or
                property_name.lower() in element.id.lower() or
                'drillhole' in element.id.lower() and 'lithology' in property_name.lower()):

                # Update the element
                element.categories = list(categories)
                if category_colors:
                    element.category_colors = dict(category_colors)
                    # Ensure all categories have visibility state
                    for cat in categories:
                        if cat not in element.category_visible:
                            element.category_visible[cat] = True
                logger.info(f"Synced multi-mode element '{element.id}': {len(categories)} categories, {len(category_colors)} colors")

        self.widget.update()

    def _normalize_category_colors(
        self,
        colors: Dict[Union[str, int, float], Any],
    ) -> Dict[Union[str, int, float], Tuple[float, float, float, float]]:
        """Convert assorted color representations to RGBA float tuples."""
        normalized: Dict[Union[str, int, float], Tuple[float, float, float, float]] = {}
        for key, value in colors.items():
            rgba: Optional[Tuple[float, float, float, float]] = None
            try:
                if isinstance(value, QColor):
                    rgba = (value.redF(), value.greenF(), value.blueF(), value.alphaF())
                elif isinstance(value, str):
                    qc = QColor(value)
                    if qc.isValid():
                        rgba = (qc.redF(), qc.greenF(), qc.blueF(), qc.alphaF())
                elif isinstance(value, (tuple, list, np.ndarray)):
                    arr = np.asarray(value, dtype=float).flatten()
                    if arr.size >= 3:
                        if np.any(arr > 1.0001):
                            arr = arr / 255.0
                        if arr.size == 3:
                            arr = np.append(arr, 1.0)
                        rgba = tuple(float(np.clip(c, 0.0, 1.0)) for c in arr[:4])
                elif isinstance(value, dict):
                    # Support dicts with r/g/b[/a] keys
                    r = value.get("r", value.get("red", 0))
                    g = value.get("g", value.get("green", 0))
                    b = value.get("b", value.get("blue", 0))
                    a = value.get("a", value.get("alpha", 1))
                    rgba = self._normalize_category_colors({key: (r, g, b, a)}).get(key)
            except Exception:
                rgba = None
            if rgba is None:
                rgba = (0.5, 0.5, 0.5, 1.0)
            normalized[key] = rgba
        return normalized

    def update_geology_legend(
        self,
        unit_names: List[str],
        colors: List[str],
        layer_name: str = "Geological Units"
    ) -> None:
        """
        Specifically updates the legend for Geological Solids/Surfaces.
        
        This method ensures the legend matches the colors used by the renderer
        for geological units, providing instant visual feedback when the
        geological model is loaded.
        
        Args:
            unit_names: List of geological unit names (e.g., ['Unit_A', 'Unit_B'])
            colors: List of hex color strings matching the units
            layer_name: Title for the legend (default: "Geological Units")
        """
        if not unit_names:
            return
        
        # Convert hex strings to RGBA float tuples for the widget
        category_colors: Dict[Union[str, int, float], Tuple[float, float, float, float]] = {}
        for name, hex_color in zip(unit_names, colors):
            try:
                qc = QColor(hex_color)
                if qc.isValid():
                    category_colors[name] = (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
                else:
                    category_colors[name] = (0.5, 0.5, 0.5, 1.0)
            except Exception:
                category_colors[name] = (0.5, 0.5, 0.5, 1.0)
        
        logger.info(f"Updating geology legend: {len(unit_names)} units, title='{layer_name}'")
        
        # Force a discrete legend update
        self.update_discrete(
            property_name=layer_name,
            categories=unit_names,
            category_colors=category_colors,
            subtitle="Geological Units"
        )
        
        # Ensure widget is visible
        if self.widget:
            try:
                self.widget.show()
                self.widget.update()
                self.widget.raise_()
            except Exception:
                pass
    
    def force_colormap(self, cmap_name: str) -> None:
        if self.widget is None or self._current_property is None:
            return
        if self.widget.config.type == LegendType.CONTINUOUS and self._current_data is not None:
            self.update_continuous(self._current_property, self._current_data, cmap_name=cmap_name)
        elif self.widget.config.type == LegendType.DISCRETE and self.widget.config.categories:
            self.update_discrete(self._current_property, self.widget.config.categories, cmap_name=cmap_name)

    def show_empty(self, message: str = "No data") -> None:
        if self.widget is None:
            # Queue a minimal empty payload for later binding
            payload = LegendPayload(
                layer=None,
                property=None,
                title="",
                mode="discrete",
                colormap="tab20",
                categories=[],
                category_colors={},
                subtitle=message,
            )
            self._pending_payloads.append(payload)
            self._current_data = None
            self._current_property = None
            return

        # Display an empty state message in the widget and make it visible.
        try:
            try:
                self.widget.set_empty_state(message)
            except Exception:
                # Fallback: clear the widget
                try:
                    self.widget.clear()
                except Exception:
                    pass
            if not self.widget.isVisible():
                try:
                    self.widget.show()
                    self.widget.raise_()
                except Exception:
                    pass
            self.widget.update()
            # Animate into view for a better UX
            try:
                self._fade_in()
            except Exception:
                pass
            self._log_debug("set_visibility(show)")
        except Exception:
            pass

    def clear(self) -> None:
        """
        FIX CS-004: Reset legend state on scene clear.
        
        Clears all cached data, pending payloads, and hides the widget.
        Called when the scene is cleared or the application returns to EMPTY state.
        """
        # Clear cached data
        self._current_data = None
        self._current_property = None
        self._categories = None
        self._vmin = None
        self._vmax = None
        self._current_payload = None
        self._pending_payloads.clear()
        self._current_namespace = None
        
        # Reset visibility
        self._visible = False
        
        # Clear and hide widget
        if self.widget is not None:
            try:
                self.widget.set_empty_state("No data")
                self.widget.hide()
            except Exception:
                pass
        
        self._log_debug("clear() - legend state reset")

    def set_orientation(self, orientation: str) -> None:
        if self.widget is None:
            return
        self.widget.set_orientation(orientation)
        self.widget.update()
        self._state.orientation = orientation
        self._notify_state_changed()
        self._ensure_position()

    def _on_widget_orientation_changed(self, orientation: str) -> None:
        """
        Respond to orientation changes emitted by the LegendWidget.

        The legend widget allows the user to toggle between vertical and
        horizontal orientations via its context menu. When the user does
        so, it emits an ``orientation_changed`` signal. If the
        LegendManager doesn’t handle this signal, it will continue to
        believe the orientation is unchanged and will reset it on the next
        update. This handler simply forwards the new orientation to
        ``set_orientation`` so the manager’s state remains in sync with
        the widget.
        """
        try:
            self.set_orientation(orientation)
        except Exception:
            # Do not propagate exceptions back to the Qt event loop.
            pass

    def set_font_size(self, size: int) -> None:
        if self.widget is None:
            return
        self.widget.set_theme(font_size=int(size))
        self.widget.update()
        self._ensure_position()

    def apply_style(self, style_dict: Dict[str, Any]) -> None:
        if self.widget is None:
            return

        orientation = style_dict.get("orientation")
        if orientation:
            self.widget.set_orientation(orientation)
            self._state.orientation = orientation

        font_size = style_dict.get("font_size")
        if font_size:
            self.widget.set_theme(font_size=int(font_size))

        background = style_dict.get("background")
        opacity = style_dict.get("background_opacity")
        if background is not None or opacity is not None:
            try:
                r, g, b = background or getattr(self.renderer, "legend_background_color", (0.12, 0.12, 0.12))
                qcolor = QColor(int(r * 255), int(g * 255), int(b * 255))
                qcolor.setAlphaF(float(opacity if opacity is not None else getattr(self.renderer, "legend_background_opacity", 0.6)))
                self.widget.config.background_color = qcolor
                self._custom_background = QColor(qcolor)
            except Exception:
                pass

        if "shadow" in style_dict:
            self.widget.config.enable_shadow = bool(style_dict["shadow"])
        if "outline" in style_dict:
            self.widget.config.draw_outline = bool(style_dict["outline"])
        if "count" in style_dict:
            self.widget.config.tick_count = max(2, int(style_dict["count"]))
        if "decimals" in style_dict:
            self.widget.config.label_decimals = int(style_dict["decimals"])

        self.widget.update()
        self._ensure_position()
        self._notify_state_changed()

    # ------------------------------------------------------------------ #
    # Convenience setters used by SceneManager / SyncHub
    # ------------------------------------------------------------------ #

    def set_continuous(
        self,
        field: str,
        vmin: float,
        vmax: float,
        low_colour=None,
        high_colour=None,
        cmap_name: str = None
    ) -> None:
        """
        Set a continuous legend. Colours rendered by widget/renderer.

        Args:
            field: Property name
            vmin: Minimum value
            vmax: Maximum value
            low_colour: Optional low color override
            high_colour: Optional high color override
            cmap_name: Colormap name (e.g., 'viridis', 'turbo'). If None, uses existing colormap.
        """
        self._current_property = field
        self._categories = None
        self._vmin = float(vmin) if vmin is not None else None
        self._vmax = float(vmax) if vmax is not None else None

        # Update colormap if provided
        if cmap_name is not None:
            self._colormap = cmap_name

        payload = {
            "property": field,
            "title": field,
            "vmin": self._vmin,
            "vmax": self._vmax,
            "colormap": self._colormap,
            "categories": None,
            "mode": "continuous",
            "low_colour": low_colour,
            "high_colour": high_colour,
        }
        self.legend_changed.emit(payload)

        # Store as LegendPayload for state capture
        self._current_payload = LegendPayload(
            layer=None,
            property=field,
            title=field,
            mode="continuous",
            colormap=self._colormap or "viridis",
            vmin=self._vmin,
            vmax=self._vmax,
        )

    def set_discrete_bins(self, field: str, bins: list, colours: list) -> None:
        """Set discrete bin legend."""
        self._current_property = field

        # categories format used by LegendWidget
        self._categories = [
            {"label": f"{bins[i]}–{bins[i+1]}", "colour": colours[i]}
            for i in range(len(colours))
        ]

        payload = {
            "property": field,
            "title": field,
            "vmin": None,
            "vmax": None,
            "colormap": None,
            "categories": self._categories,
            "mode": "discrete",
            "bins": bins,
            "colours": colours,
        }
        self.legend_changed.emit(payload)

        # Store as LegendPayload for state capture
        # Convert category list to dict format for LegendPayload
        category_colors = {}
        for cat_info in self._categories:
            label = cat_info.get("label", "")
            colour = cat_info.get("colour", (0.5, 0.5, 0.5))
            # Ensure RGBA format
            if len(colour) == 3:
                colour = (*colour, 1.0)
            category_colors[label] = colour

        self._current_payload = LegendPayload(
            layer=None,
            property=field,
            title=field,
            mode="discrete",
            colormap="custom",
            categories=[cat["label"] for cat in self._categories],
            category_colors=category_colors,
        )

    def set_lithology_lut(self, field: str, lut: dict) -> None:
        """Set lithology legend from LUT {code: colour}."""
        self._current_property = field
        self._categories = [
            {"label": k, "colour": v} for k, v in (lut or {}).items()
        ]

        payload = {
            "property": field,
            "title": field,
            "vmin": None,
            "vmax": None,
            "colormap": None,
            "categories": self._categories,
            "mode": "lithology",
            "lut": lut,
        }
        self.legend_changed.emit(payload)

        # Store as LegendPayload for state capture
        category_colors = {}
        for cat_info in self._categories:
            label = cat_info.get("label", "")
            colour = cat_info.get("colour", (0.5, 0.5, 0.5))
            # Ensure RGBA format
            if len(colour) == 3:
                colour = (*colour, 1.0)
            category_colors[label] = colour

        self._current_payload = LegendPayload(
            layer=None,
            property=field,
            title=field,
            mode="discrete",  # lithology is a type of discrete legend
            colormap="lithology",
            categories=[cat["label"] for cat in self._categories],
            category_colors=category_colors,
        )

    def set_ore_waste(self, field: str) -> None:
        """Standard ore / waste legend."""
        lut = {
            "Ore": (0.80, 0.00, 0.00),
            "Waste": (0.50, 0.50, 0.50)
        }
        self.set_custom_lut(field, lut)

    def set_custom_lut(self, field: str, lut: dict) -> None:
        """Set custom discrete legend."""
        self._current_property = field
        self._categories = [
            {"label": k, "colour": v} for k, v in (lut or {}).items()
        ]

        payload = {
            "property": field,
            "title": field,
            "vmin": None,
            "vmax": None,
            "colormap": None,
            "categories": self._categories,
            "mode": "custom",
            "lut": lut,
        }
        self.legend_changed.emit(payload)

        # Store as LegendPayload for state capture
        category_colors = {}
        for cat_info in self._categories:
            label = cat_info.get("label", "")
            colour = cat_info.get("colour", (0.5, 0.5, 0.5))
            # Ensure RGBA format
            if len(colour) == 3:
                colour = (*colour, 1.0)
            category_colors[label] = colour

        self._current_payload = LegendPayload(
            layer=None,
            property=field,
            title=field,
            mode="discrete",  # custom is a type of discrete legend
            colormap="custom",
            categories=[cat["label"] for cat in self._categories],
            category_colors=category_colors,
        )

    def update_from_sync(self, st: dict) -> None:
        """Entry point for SyncHub legend propagation."""
        mode = st.get("mode")

        if mode == "continuous":
            self.set_continuous(
                st.get("field", ""),
                st.get("vmin"),
                st.get("vmax"),
                st.get("low"),
                st.get("high"),
            )

        elif mode == "discrete":
            self.set_discrete_bins(
                st.get("field", ""),
                st.get("bins", []),
                st.get("colours", []),
            )

        elif mode == "lithology":
            self.set_lithology_lut(
                st.get("field", ""),
                st.get("lut", {}),
            )

        elif mode == "ore_waste":
            self.set_ore_waste(st.get("field", ""))

        elif mode == "custom":
            self.set_custom_lut(
                st.get("field", ""),
                st.get("lut", {}),
            )

    def set_visibility(self, visible: bool) -> None:
        """Compatibility shim: set legend visibility on the manager.

        Some callers (renderer) call `set_visibility`; ensure this exists and
        delegates to the widget's show/hide behavior while updating manager
        state.
        """
        try:
            self._visible = bool(visible)
            if self.widget is None:
                return
            if visible:
                try:
                    self.widget.show()
                    self.widget.raise_()
                except Exception:
                    pass
                self._log_debug("set_visibility(show)")
            else:
                try:
                    self.widget.hide()
                except Exception:
                    pass
                self._log_debug("set_visibility(hide)")
            self._notify_state_changed()
            # Emit visibility changed signal for UI synchronization
            self.visibility_changed.emit(self._visible)
        except Exception:
            # Never let visibility toggles raise into the renderer
            pass

    def get_state(self) -> Dict[str, Any]:
        """Get current legend state as a dictionary.

        Returns:
            Dict containing legend state including visibility, position, size, etc.
        """
        return self._state_as_dict()

    # ---------------------------------------------------------------------#
    # Internals
    # ---------------------------------------------------------------------#
    def _apply_payload(self, payload: LegendPayload, *, animate: bool) -> None:
        if self.widget is None:
            return

        self._apply_theme_from_renderer()

        if payload.mode == "discrete":
            self.update_discrete(payload.title, payload.categories, category_colors=payload.category_colors, cmap_name=payload.colormap, subtitle=payload.subtitle)
        else:
            data = payload.data if payload.data is not None else self._current_data
            if data is None:
                # Without raw data we fallback to the mapper range
                synthetic = np.linspace(payload.vmin or 0.0, payload.vmax or 1.0, 256)
                self.update_continuous(payload.title, synthetic, cmap_name=payload.colormap, log_scale=payload.log_scale, subtitle=payload.subtitle)
                self._current_data = synthetic
            else:
                self.update_continuous(
                    payload.title,
                    data,
                    cmap_name=payload.colormap,
                    log_scale=payload.log_scale,
                    subtitle=payload.subtitle,
                )
                if payload.vmin is not None:
                    self.widget.config.vmin = float(payload.vmin)
                if payload.vmax is not None:
                    self.widget.config.vmax = float(payload.vmax)

        orientation = self._state.orientation or getattr(self.renderer, "legend_position", "vertical")
        try:
            self.widget.set_orientation(orientation)
        except Exception:
            pass
        self._state.orientation = orientation

        self._ensure_position()

        if self._visible:
            self.widget.show()
            if animate:
                # Ensure widget is immediately visible before starting an animation
                try:
                    self.widget.setWindowOpacity(1.0)
                except Exception:
                    pass
                self._fade_in()
            else:
                try:
                    self.widget.setWindowOpacity(1.0)
                except Exception:
                    pass
            try:
                self.widget.raise_()
            except Exception:
                pass
            self._log_debug("payload_applied", {"animate": animate})

        if self.renderer is not None:
            self.renderer._legend_context = {"layer": payload.layer, "property": payload.property}

    def _fade_in(self) -> None:
        if self.widget is None or self._animation is None:
            return
        try:
            self._animation.stop()
            # Start from a small but visible opacity so the widget is never
            # completely transparent while the animation runs. Some platforms
            # treat fully-zero opacity specially which can make the legend
            # appear to be missing briefly in logs and on-screen.
            start_opacity = 0.05
            try:
                self.widget.setWindowOpacity(start_opacity)
            except Exception:
                # If the platform doesn't support window opacity, fall back
                # to the animation values without pre-setting the widget.
                start_opacity = 0.0
            self._animation.setStartValue(start_opacity)
            self._animation.setEndValue(1.0)
            self._animation.start()
        except Exception:
            pass

    def _gather_renderer_state(self) -> Optional[LegendPayload]:
        renderer = self.renderer
        if renderer is None:
            return None

        # Fallbacks for missing layer/property
        layer_name = getattr(renderer, '_get_active_layer_name', lambda: None)()
        if not layer_name or layer_name not in renderer.active_layers:
            # Use first available layer if possible
            if renderer.active_layers:
                layer_name = next(iter(renderer.active_layers.keys()))
            else:
                return None

        # DRILLHOLE SPECIAL HANDLING: Use pre-computed legend metadata
        # Drillholes have multiple actors and special data structure, so we use
        # the metadata computed by _update_drillhole_legend_fast instead
        if layer_name == "drillholes" or "drillhole" in layer_name.lower():
            drillhole_metadata = getattr(renderer, '_drillhole_legend_metadata', None)
            if drillhole_metadata:
                mode = drillhole_metadata.get('mode', 'discrete')
                prop_name = drillhole_metadata.get('property', 'Lithology')
                colormap = drillhole_metadata.get('colormap', 'tab10')
                categories = drillhole_metadata.get('categories', [])
                category_colors = drillhole_metadata.get('category_colors', {})
                vmin = drillhole_metadata.get('vmin')
                vmax = drillhole_metadata.get('vmax')

                # Build data array for continuous mode
                data = None
                if mode == 'continuous' and vmin is not None and vmax is not None:
                    data = np.array([vmin, vmax], dtype=np.float32)

                return LegendPayload(
                    layer=layer_name,
                    property=prop_name,
                    title=prop_name,
                    mode=mode,
                    colormap=colormap,
                    vmin=float(vmin) if vmin is not None else None,
                    vmax=float(vmax) if vmax is not None else None,
                    log_scale=False,
                    reverse=False,
                    data=data,
                    categories=categories,
                    category_colors=category_colors,
                    subtitle="",
                )
            else:
                # No drillhole metadata yet - return None to avoid "No data"
                # The legend will be updated when drillholes are rendered
                logger.debug("Drillhole layer found but no legend metadata available yet")
                return None

        prop_name = getattr(renderer, '_get_active_property_for_layer', lambda x: None)(layer_name)
        if not prop_name:
            # Use first available property if possible
            layer = renderer.active_layers[layer_name]
            props = getattr(layer, 'properties', None)
            if props and isinstance(props, dict):
                prop_name = next(iter(props.keys()))
            else:
                prop_name = 'ZN_est'  # fallback to ZN_est if present

        layer = renderer.active_layers[layer_name]
        actor = getattr(layer, 'actor', None)
        if actor is None and layer_name == "Block Model":
            actor = getattr(renderer, 'mesh_actor', None)
        if actor is None:
            return None

        try:
            mapper = actor.GetMapper()
            vmin, vmax = mapper.GetScalarRange()
        except Exception:
            mapper = None
            vmin = vmax = None

        if mapper is None or not np.all(np.isfinite([vmin, vmax])):
            # Fallback to default range
            vmin, vmax = 0.0, 1.0

        data_ref = layer.get("data") if hasattr(layer, 'get') else None
        data = None
        try:
            if layer_name == "Block Model" and getattr(renderer, 'current_model', None) is not None:
                data = renderer.current_model.get_property(prop_name)
            elif isinstance(data_ref, dict) and "mesh" in data_ref:
                mesh = data_ref.get("mesh")
                if mesh is not None:
                    if hasattr(mesh, "cell_data") and prop_name in mesh.cell_data:
                        data = mesh.cell_data[prop_name]
                    elif hasattr(mesh, "point_data") and prop_name in mesh.point_data:
                        data = mesh.point_data[prop_name]
            elif data_ref is not None:
                if hasattr(data_ref, "cell_data") and prop_name in data_ref.cell_data:
                    data = data_ref.cell_data[prop_name]
                elif hasattr(data_ref, "point_data") and prop_name in data_ref.point_data:
                    data = data_ref.point_data[prop_name]
        except Exception as exc:
            logger.debug("Failed to extract legend data: %s", exc)
            data = None

        # Fallback to default data if missing
        if data is None:
            data = np.linspace(vmin, vmax, 6)

        colormap = getattr(renderer, 'current_colormap', None) or "viridis"
        mode = getattr(renderer, 'current_color_mode', None) or "continuous"
        subtitle = getattr(renderer, "legend_subtitle", "")
        custom_colors = getattr(renderer, 'current_custom_colors', None)

        # Build categorical colours if necessary
        categories: List[Union[str, int, float]] = []
        category_colors: Dict[Union[str, int, float], Tuple[float, float, float, float]] = {}

        if mode == "discrete":
            data_arr = np.asarray(data) if data is not None else None
            if custom_colors:
                categories = list(custom_colors.keys())
                for key, value in custom_colors.items():
                    try:
                        rgba = mcolors.to_rgba(value)
                    except Exception:
                        rgba = (0.5, 0.5, 0.5, 1.0)
                    category_colors[key] = tuple(float(v) for v in rgba)
            elif data_arr is not None:
                if np.issubdtype(data_arr.dtype, np.number):
                    finite = data_arr[np.isfinite(data_arr)]
                    unique = np.unique(finite)
                else:
                    arr = data_arr.astype(str)
                    unique = np.unique(arr[arr != ""])
                categories = unique.tolist()
                cmap_obj = cm.get_cmap(colormap or "tab20")
                for idx, cat in enumerate(categories):
                    sample = cmap_obj(idx / max(1, len(categories) - 1))
                    category_colors[cat] = (float(sample[0]), float(sample[1]), float(sample[2]), float(sample[3]))

        title = f"{layer_name}: {prop_name}" if layer_name else prop_name

        return LegendPayload(
            layer=layer_name,
            property=prop_name,
            title=title,
            mode=mode,
            colormap=colormap,
            vmin=float(vmin) if vmin is not None else None,
            vmax=float(vmax) if vmax is not None else None,
            log_scale=getattr(self.widget.config, "log_scale", False) if self.widget else False,
            reverse=getattr(self.widget.config, "reverse", False) if self.widget else False,
            data=np.asarray(data) if data is not None else None,
            categories=categories,
            category_colors=category_colors,
            subtitle=subtitle,
        )

    def _apply_theme_from_renderer(self) -> None:
        if self.widget is None or self.renderer is None:
            return

        font_size = getattr(self.renderer, "legend_font_size", None)
        if font_size:
            try:
                self.widget.config.font_size = int(font_size)
            except Exception:
                pass

        if self._custom_background is not None:
            base_bg = QColor(self._custom_background)
            self._apply_widget_palette(base_bg)
            return

        background = getattr(self.renderer, "background_color", "#1f1f1f")
        scene_color = _to_qcolor(background)
        luminance = _relative_luminance(scene_color)

        if luminance > 0.55:
            base_bg = QColor(24, 28, 36, 215)
        else:
            base_bg = QColor(248, 248, 252, 220)

        self._apply_widget_palette(base_bg)

    def _apply_widget_palette(self, base_bg: QColor) -> None:
        if self.widget is None:
            return
        color = QColor(base_bg)
        text = _contrasting_text_color(color)
        border = QColor(color)
        border.setAlpha(max(80, int(color.alpha() * 0.9)))

        self.widget.config.background_color = color
        self.widget.config.text_color = text
        self.widget.config.tick_color = QColor(text)
        self.widget.config.border_color = border
        self.widget.config.bar_outline_color = border
        try:
            self.widget.update()
        except Exception:
            pass
        self._state.background_rgba = (color.red(), color.green(), color.blue(), color.alpha())
        self._log_debug("theme_applied", {"color": (color.red(), color.green(), color.blue(), color.alpha())})

    # ---------------------------------------------------------------------#
    # Utility
    # ---------------------------------------------------------------------#

    def _on_background_color_changed(self, color: QColor) -> None:
        if not isinstance(color, QColor) or not color.isValid():
            return
        self._custom_background = QColor(color)
        if self.renderer is not None:
            try:
                self.renderer.legend_background_color = (
                    color.red() / 255.0,
                    color.green() / 255.0,
                    color.blue() / 255.0,
                )
                self.renderer.legend_background_opacity = color.alphaF()
            except Exception:
                pass
        self._apply_widget_palette(color)
        if self.widget is not None:
            self.widget.update()
        self._state.background_rgba = (color.red(), color.green(), color.blue(), color.alpha())
        self._notify_state_changed()
        self._log_debug("background_changed", {"color": (color.red(), color.green(), color.blue(), color.alpha())})

    def _on_dock_requested(self, anchor: str) -> None:
        if anchor not in {"floating", "top_left", "top_right", "bottom_left", "bottom_right"}:
            return
        self._state.anchor = anchor
        if anchor != "floating":
            # Snap to anchor with current margin
            self._state.position = self._compute_anchor_position(anchor)
        if self.widget is not None:
            try:
                self.widget.set_current_anchor(anchor)
                self._ensure_position()
            except Exception:
                pass
        self._notify_state_changed()

    def _on_floating_position_changed(self, x: int, y: int) -> None:
        self._state.position = (int(x), int(y))
        if self._state.anchor != "floating":
            self._state.anchor = "floating"
            if self.widget is not None:
                try:
                    self.widget.set_current_anchor("floating")
                except Exception:
                    pass
        self._notify_state_changed()

    def _on_widget_resized(self, width: int, height: int) -> None:
        self._state.size = (int(width), int(height))
        self._notify_state_changed()
    
    def _on_category_label_changed(self, category: Union[str, int, float], label: str) -> None:
        """Handle category label change from widget - persist to registry."""
        if self._registry_ref is None or self._current_namespace is None:
            logger.debug("Cannot persist category label: registry or namespace not available")
            return
        
        try:
            if label:
                # Set new label
                self._registry_ref.set_category_label(self._current_namespace, str(category), label)
            else:
                # Reset label (empty string means reset)
                self._registry_ref.clear_category_label(self._current_namespace, str(category))
            
            logger.info(f"Persisted category label: {self._current_namespace}.{category} = '{label}'")
        except Exception as e:
            logger.error(f"Failed to persist category label: {e}", exc_info=True)
    
    def _on_category_color_changed(self, category: Union[str, int, float], color: Tuple[float, float, float, float]) -> None:
        """Handle category color change from widget - update renderer meshes for ALL layer types."""
        logger.info(f"LegendManager: Category color changed: {category} -> {color}")

        try:
            if not self.renderer:
                logger.debug("No renderer available for color update")
                return

            updated = False
            color_hex = self._rgba_to_hex(color)

            # 1. Try geology surface/solid layers
            if hasattr(self.renderer, 'update_geology_layer_color'):
                geo_updated = self.renderer.update_geology_layer_color(str(category), color)
                if geo_updated:
                    logger.info(f"Updated geology mesh color for domain '{category}'")
                    updated = True

            # 2. Try classification layers
            if hasattr(self.renderer, 'active_layers') and hasattr(self.renderer, '_update_classification_colors'):
                for layer_name, layer_info in self.renderer.active_layers.items():
                    layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
                    if layer_type == 'classification':
                        layer_data = layer_info.get('data', {})
                        if isinstance(layer_data, dict) and 'mesh' in layer_data:
                            mesh = layer_data['mesh']
                            # Build custom colors dict with this category
                            custom_colors = {str(category): color_hex}
                            try:
                                self.renderer._update_classification_colors(
                                    layer_name, layer_info, mesh,
                                    colormap='custom', color_mode='discrete',
                                    custom_colors=custom_colors
                                )
                                logger.info(f"Updated classification layer '{layer_name}' color for '{category}'")
                                updated = True
                            except Exception as e:
                                logger.warning(f"Failed to update classification layer '{layer_name}': {e}")

            # 3. Try block model layers with discrete coloring
            if hasattr(self.renderer, 'active_layers') and hasattr(self.renderer, 'update_layer_property'):
                for layer_name, layer_info in self.renderer.active_layers.items():
                    layer_type = layer_info.get('layer_type', layer_info.get('type', ''))
                    if 'block' in layer_name.lower() or layer_type == 'block_model':
                        # Get current property and colormap
                        current_prop = layer_info.get('current_property', layer_info.get('property', 'Formation'))
                        current_cmap = layer_info.get('colormap', 'tab10')

                        # Build custom colors dict
                        custom_colors = {str(category): color_hex}
                        try:
                            self.renderer.update_layer_property(
                                layer_name, current_prop, current_cmap, 'discrete',
                                custom_colors=custom_colors
                            )
                            logger.info(f"Updated block model '{layer_name}' color for category '{category}'")
                            updated = True
                        except Exception as e:
                            logger.warning(f"Failed to update block model '{layer_name}': {e}")

            # 4. Try drillhole layers (lithology coloring)
            if hasattr(self.renderer, 'active_layers'):
                drillhole_layer = self.renderer.active_layers.get('drillholes')
                if drillhole_layer:
                    layer_data = drillhole_layer.get('data', {})
                    if isinstance(layer_data, dict):
                        lith_colors = layer_data.get('lith_colors', {})
                        # Check if category matches a lithology code
                        cat_str = str(category)
                        if cat_str in lith_colors or category in lith_colors:
                            # Update the lithology color
                            rgb = (color[0], color[1], color[2])
                            lith_colors[cat_str] = rgb
                            if category != cat_str:
                                lith_colors[category] = rgb
                            layer_data['lith_colors'] = lith_colors

                            # Trigger drillhole re-render with new colors
                            if hasattr(self.renderer, '_update_drillhole_colors'):
                                try:
                                    current_prop = layer_data.get('current_property', 'Lithology')
                                    current_cmap = layer_data.get('colormap', 'tab10')
                                    self.renderer._update_drillhole_colors(
                                        layer_data, current_prop, current_cmap, 'discrete'
                                    )
                                    logger.info(f"Updated drillhole lithology color for '{category}'")
                                    updated = True
                                except Exception as e:
                                    logger.warning(f"Failed to update drillhole colors: {e}")

            # 5. Force render if any updates were made
            if updated and hasattr(self.renderer, 'plotter') and self.renderer.plotter:
                try:
                    self.renderer.plotter.render()
                    logger.debug("Rendered after category color update")
                except Exception as e:
                    logger.debug(f"Render after color update failed: {e}")

            if not updated:
                logger.debug(f"No renderer layers found for category '{category}'")

        except Exception as e:
            logger.error(f"Failed to update mesh color for category '{category}': {e}", exc_info=True)

    def _rgba_to_hex(self, color: Tuple[float, float, float, float]) -> str:
        """Convert RGBA tuple (0-1 range) to hex color string."""
        r = int(color[0] * 255)
        g = int(color[1] * 255)
        b = int(color[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _on_label_maps_changed(self, namespace: str) -> None:
        """Handle label map changes from registry - refresh widget if it's the current namespace."""
        if namespace != self._current_namespace:
            return
        
        # Refresh labels from registry
        if self.widget and self._registry_ref:
            try:
                self._apply_category_labels_from_registry()
                self.widget.update()
                logger.debug(f"Refreshed category labels for namespace: {namespace}")
            except Exception as e:
                logger.error(f"Failed to refresh category labels: {e}", exc_info=True)

    def _ensure_position(self) -> None:
        """Place the legend near the top-right corner of its parent widget."""
        widget = self.widget
        if widget is None:
            return
        parent = widget.parentWidget()
        if parent is None:
            return
        try:
            widget.adjustSize()
        except Exception:
            pass
        try:
            size_override = (
                self._state.size if self._state.size else (widget.width(), widget.height())
            )
            widget.apply_layout(
                anchor=self._state.anchor,
                position=self._state.position,
                margin=self._state.margin,
                size=size_override,
            )
            self._state.size = (widget.width(), widget.height())
            self._state.position = (widget.x(), widget.y())
        except Exception:
            pass
        try:
            widget.raise_()
        except Exception:
            pass
        self._log_debug(
            "ensure_position",
            {
                "widget_pos": (widget.x(), widget.y()),
                "widget_size": (widget.width(), widget.height()),
                "parent_size": (parent.width(), parent.height()) if parent is not None else None,
                "anchor": self._state.anchor,
            },
        )
        self._notify_state_changed()

    def _log_debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            widget = self.widget
            parent = widget.parentWidget() if widget is not None else None
            info: Dict[str, Any] = {
                "message": message,
                "visible": widget.isVisible() if widget is not None else None,
                "opacity": widget.windowOpacity() if widget is not None else None,
                "pos": (widget.x(), widget.y()) if widget is not None else None,
                "size": (widget.width(), widget.height()) if widget is not None else None,
                "parent_size": (parent.width(), parent.height()) if parent is not None else None,
                "parent": parent.metaObject().className() if parent is not None else None,
            }
            if extra:
                info.update(extra)
            logger.debug(f"LegendManager: {info}")
        except Exception:
            pass

    # ------------------------------------------------------------------#
    # State management helpers
    # ------------------------------------------------------------------#
    def _notify_state_changed(self) -> None:
        if self.renderer is None:
            return
        try:
            self.renderer.on_legend_state_changed(self._state_as_dict())
        except Exception:
            pass

    def _state_as_dict(self) -> Dict[str, Any]:
        """Get complete legend state including both UI positioning and content."""
        state = {
            "anchor": self._state.anchor,
            "position": list(self._state.position),
            "size": list(self._state.size),
            "margin": int(self._state.margin),
            "visible": bool(self._visible),
            "orientation": self._state.orientation,
            "background_rgba": list(self._state.background_rgba),
        }

        # Include current payload data (property, colormap, values, categories)
        # This ensures layouts capture the actual legend content, not just positioning
        if self._current_payload is not None:
            payload = self._current_payload
            state["property"] = payload.property
            state["title"] = payload.title
            state["mode"] = payload.mode
            state["colormap"] = payload.colormap

            if payload.vmin is not None:
                state["vmin"] = float(payload.vmin)
            if payload.vmax is not None:
                state["vmax"] = float(payload.vmax)

            state["log_scale"] = payload.log_scale
            state["reverse"] = payload.reverse

            if payload.categories:
                state["categories"] = payload.categories
            if payload.category_colors:
                # Convert category colors to serializable format
                state["category_colors"] = {
                    str(k): list(v) for k, v in payload.category_colors.items()
                }

            logger.info(f"[STATE CAPTURE] Included payload: property={payload.property}, colormap={payload.colormap}, vmin={payload.vmin}, vmax={payload.vmax}")
        else:
            logger.warning("[STATE CAPTURE] No current payload available - legend content will be missing!")

        return state

    def _load_state(self, state: Dict[str, Any]) -> None:
        try:
            anchor = state.get("anchor", self._state.anchor)
            if anchor in {"floating", "top_left", "top_right", "bottom_left", "bottom_right"}:
                self._state.anchor = anchor
            pos = state.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                self._state.position = (int(pos[0]), int(pos[1]))
            size = state.get("size")
            if isinstance(size, (list, tuple)) and len(size) == 2:
                self._state.size = (max(int(size[0]), 120), max(int(size[1]), 120))
            margin = state.get("margin")
            if isinstance(margin, (int, float)):
                self._state.margin = int(max(0, margin))
            orientation = state.get("orientation")
            if orientation in {"vertical", "horizontal"}:
                self._state.orientation = orientation
            visible = state.get("visible")
            if isinstance(visible, bool):
                self._state.visible = visible
                self._visible = visible
            background = state.get("background_rgba")
            if isinstance(background, (list, tuple)) and len(background) == 4:
                rgba = tuple(int(max(0, min(255, v))) for v in background)
                self._state.background_rgba = rgba  # type: ignore[assignment]
        except Exception:
            pass

    def _compute_anchor_position(self, anchor: str) -> Tuple[int, int]:
        widget = self.widget
        parent = widget.parentWidget() if widget is not None else None
        if widget is None or parent is None:
            return self._state.position
        parent_w = max(parent.width(), 1)
        parent_h = max(parent.height(), 1)
        width = widget.width()
        height = widget.height()
        margin = self._state.margin
        if anchor == "top_left":
            return (margin, margin)
        if anchor == "top_right":
            return (max(margin, parent_w - width - margin), margin)
        if anchor == "bottom_left":
            return (margin, max(margin, parent_h - height - margin))
        if anchor == "bottom_right":
            return (max(margin, parent_w - width - margin), max(margin, parent_h - height - margin))
        return self._state.position
    
    def _determine_label_namespace(self, property_name: str) -> str:
        """
        Determine the label namespace for the current legend context.
        
        Args:
            property_name: Current property being displayed
            
        Returns:
            Namespace string for category label lookups
        """
        # Check if this is drillhole lithology
        prop_lower = property_name.lower()
        if "lithology" in prop_lower or "lith" in prop_lower:
            return "drillholes.lithology"
        
        # For other discrete legends, use layer.property format
        layer_name = getattr(self, '_current_layer', None)
        if layer_name:
            return f"{layer_name}.{property_name}"
        
        # Fallback
        return f"unknown.{property_name}"
    
    def _apply_category_labels_from_registry(self) -> None:
        """Apply category labels from registry to the widget."""
        if self.widget is None or self._registry_ref is None or self._current_namespace is None:
            return

        try:
            # Get the label maps from registry
            maps = self._registry_ref.get_category_label_maps()
            namespace_map = maps.get(self._current_namespace, {})

            # Build category_labels dict for all current categories
            category_labels = {}
            for cat in self.widget.config.categories:
                if str(cat) in namespace_map:
                    category_labels[cat] = namespace_map[str(cat)]

            # Apply to widget
            self.widget.config.category_labels = category_labels
            logger.debug(f"Applied {len(category_labels)} category labels from namespace: {self._current_namespace}")
        except Exception as e:
            logger.error(f"Failed to apply category labels from registry: {e}", exc_info=True)

    # =========================================================================
    # Multi-Element Legend Support
    # =========================================================================

    def bind_multi_widget(self, multi_widget: "MultiLegendWidget") -> None:
        """
        Attach a MultiLegendWidget for multi-element legend support.

        Args:
            multi_widget: The MultiLegendWidget instance
        """
        self._multi_widget = multi_widget

        # Connect signals
        try:
            multi_widget.add_requested.connect(self._on_add_requested)
            multi_widget.element_removed.connect(self._on_element_removed)
            multi_widget.category_toggled.connect(self._on_multi_category_toggled)
        except Exception as e:
            logger.error(f"Failed to connect multi-widget signals: {e}")

    def _on_add_requested(self) -> None:
        """Handle add element request from multi-widget."""
        # Import here to avoid circular imports
        from .legend_add_dialog import LegendAddDialog

        dialog = LegendAddDialog(self.renderer)
        dialog.item_selected.connect(self._on_dialog_item_selected)
        dialog.exec()

    def _on_dialog_item_selected(self, layer_name: str, property_name: Optional[str], is_discrete: bool):
        """Handle selection from add dialog."""
        self.add_legend_for_layer(layer_name, property_name)

    def _on_element_removed(self, element_id: str) -> None:
        """Handle element removal."""
        logger.debug(f"Legend element removed: {element_id}")

    def _on_multi_category_toggled(self, element_id: str, category: object, visible: bool) -> None:
        """Handle category visibility toggle from multi-widget."""
        # Update the visualization if needed
        try:
            parts = element_id.split('.', 1)
            if len(parts) == 2:
                layer_name, prop_name = parts
                # Notify renderer to update visibility
                if hasattr(self.renderer, 'set_category_visibility'):
                    self.renderer.set_category_visibility(layer_name, prop_name, category, visible)
        except Exception as e:
            logger.debug(f"Failed to sync category visibility: {e}")

    def add_legend_for_layer(
        self,
        layer_name: str,
        property_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Add a legend element for a specific layer/property.

        Args:
            layer_name: Name of the layer
            property_name: Property name (optional for geology layers)

        Returns:
            Element ID if successful, None otherwise
        """
        if self.widget is None:
            logger.warning("No legend widget attached, cannot add legend element")
            return None

        if self.renderer is None:
            logger.warning("No renderer attached, cannot add legend element")
            return None

        # Build element from layer data
        element = self._build_element_from_layer(layer_name, property_name)
        if element is None:
            logger.warning(f"Failed to build legend element for {layer_name}.{property_name}")
            return None

        # Add to legend widget (enables multi-mode if not already)
        self.widget.add_element(element)

        logger.info(f"Added legend element: {element.id} ({element.element_type.value})")
        return element.id

    def remove_legend_element(self, element_id: str) -> bool:
        """
        Remove a legend element.

        Args:
            element_id: The element ID to remove

        Returns:
            True if removed, False if not found
        """
        if self.widget is None:
            return False

        return self.widget.remove_element(element_id)

    def _build_element_from_layer(
        self,
        layer_name: str,
        property_name: Optional[str],
    ) -> Optional[LegendElement]:
        """
        Build a LegendElement from renderer layer data.

        Args:
            layer_name: Name of the layer
            property_name: Property name (optional)

        Returns:
            LegendElement or None if layer not found
        """
        if self.renderer is None or not hasattr(self.renderer, 'active_layers'):
            logger.warning("Cannot build element: renderer or active_layers not available")
            return None

        layer_info = self.renderer.active_layers.get(layer_name)
        if layer_info is None:
            logger.warning(f"Layer '{layer_name}' not found in active_layers")
            return None

        layer_type = layer_info.get('type', 'unknown')
        data = layer_info.get('data')

        # For block models, extract property name from layer name if not provided
        # Layer names are formatted as "Block Model: PropertyName"
        if property_name is None and ('block' in layer_type.lower() or 'Block Model' in layer_name):
            if ':' in layer_name:
                property_name = layer_name.split(':', 1)[1].strip()
                logger.debug(f"Extracted property name '{property_name}' from layer name '{layer_name}'")

        logger.debug(f"Building element for layer '{layer_name}', type='{layer_type}', property='{property_name}', has data={data is not None}")

        # Determine if discrete or continuous
        is_discrete = self._detect_discrete_layer(layer_name, property_name, layer_info)
        logger.debug(f"Layer '{layer_name}' detected as {'discrete' if is_discrete else 'continuous'}")

        # Build title - use more descriptive title for drillholes
        if property_name:
            title = property_name
        elif 'drillhole' in layer_type.lower() or 'drillholes' in layer_name.lower():
            # For drillholes, include current color property if available
            if isinstance(data, dict):
                color_mode = data.get('color_mode', 'Lithology')
                if color_mode == 'Lithology':
                    title = "Drillholes - Lithology"
                else:
                    assay_field = data.get('assay_field', 'Assay')
                    title = f"Drillholes - {assay_field}"
            else:
                title = "Drillholes"
        else:
            title = layer_name

        if is_discrete:
            # Build discrete element
            categories = self._extract_categories(layer_name, property_name, layer_info)
            logger.info(f"Extracted {len(categories)} categories for '{layer_name}': {categories[:5]}...")

            # Try to extract actual colors from layer data
            category_colors = self._extract_category_colors_from_layer(layer_info, categories)
            if not category_colors:
                # Fallback: generate colors
                logger.info(f"No colors extracted from layer, generating colors for {len(categories)} categories")
                category_colors = self._generate_category_colors(categories)
            else:
                logger.info(f"Using {len(category_colors)} colors extracted from layer data")

            element = LegendElement.create_discrete(
                layer=layer_name,
                property_name=property_name,
                title=title,
                categories=categories,
                category_colors=category_colors,
            )
            logger.info(f"Created discrete element: id='{element.id}', title='{element.title}', {len(element.categories)} categories")
            return element
        else:
            # Build continuous element
            vmin, vmax = self._extract_value_range(layer_name, property_name, layer_info)

            # Get colormap from multiple sources
            cmap_name = layer_info.get('current_colormap')
            if not cmap_name and self.renderer is not None:
                cmap_name = getattr(self.renderer, 'current_colormap', None)
            if not cmap_name:
                cmap_name = 'viridis'

            element = LegendElement.create_continuous(
                layer=layer_name,
                property_name=property_name or '',
                title=title,
                vmin=vmin,
                vmax=vmax,
                cmap_name=cmap_name,
            )
            logger.info(f"Created continuous element: id='{element.id}', title='{element.title}', vmin={vmin}, vmax={vmax}, cmap={cmap_name}")
            return element

    def _detect_discrete_layer(
        self,
        layer_name: str,
        property_name: Optional[str],
        layer_info: Dict,
    ) -> bool:
        """Detect if a layer/property should be treated as discrete."""
        layer_type = layer_info.get('type', '').lower()
        layer_name_lower = layer_name.lower()

        # Drillholes are typically discrete (lithology-colored)
        if 'drillhole' in layer_type or 'drillholes' in layer_name_lower:
            # Check if it's lithology-colored (discrete) vs assay-colored (continuous)
            data = layer_info.get('data')
            if isinstance(data, dict):
                color_mode = data.get('color_mode', 'Lithology')
                if color_mode == 'Lithology':
                    return True
                # Assay mode can be discrete if it has lith_colors
                lith_colors = data.get('lith_colors', {})
                if lith_colors:
                    return True
            return True  # Default to discrete for drillholes

        # Geology layers are always discrete
        if 'geology' in layer_type or 'surface' in layer_type:
            return True

        # Check property name for discrete keywords
        if property_name:
            prop_lower = property_name.lower()
            discrete_keywords = [
                'lithology', 'lith', 'rock', 'type', 'zone', 'domain',
                'unit', 'formation', 'class', 'category', 'code', 'name'
            ]
            for keyword in discrete_keywords:
                if keyword in prop_lower:
                    return True

        # Check data if available
        data = layer_info.get('data')
        if data is not None and property_name:
            try:
                if hasattr(data, 'point_data'):
                    prop_data = data.point_data.get(property_name)
                    if prop_data is not None:
                        # String types are discrete
                        if prop_data.dtype.kind in ('U', 'S', 'O'):
                            return True
                        # Low unique count is discrete
                        finite_mask = np.isfinite(prop_data) if prop_data.dtype.kind in ('f', 'i') else np.ones(len(prop_data), dtype=bool)
                        unique_count = len(np.unique(prop_data[finite_mask]))
                        if unique_count <= self.discrete_unique_max:
                            return True
            except Exception:
                pass

        return False

    def _extract_categories(
        self,
        layer_name: str,
        property_name: Optional[str],
        layer_info: Dict,
    ) -> List:
        """Extract category values from layer data."""
        categories = []

        # Try to get from data
        data = layer_info.get('data')
        if data is not None:
            # For drillholes, lith_colors keys are the categories
            if isinstance(data, dict) and 'lith_colors' in data:
                lith_colors = data.get('lith_colors', {})
                if lith_colors:
                    categories = list(lith_colors.keys())
                    logger.debug(f"Extracted {len(categories)} categories from drillhole lith_colors")
            elif property_name and hasattr(data, 'point_data'):
                try:
                    prop_data = data.point_data.get(property_name)
                    if prop_data is not None:
                        categories = list(np.unique(prop_data))
                except Exception:
                    pass
            elif hasattr(data, 'point_data') and 'Formation' in data.point_data:
                try:
                    categories = list(np.unique(data.point_data['Formation']))
                except Exception:
                    pass

        # Fallback: check layer metadata
        if not categories:
            categories = layer_info.get('categories', [])

        # Filter out NaN/empty
        categories = [c for c in categories if c is not None and str(c) not in ('nan', '', 'None')]

        return categories[:50]  # Limit to 50 categories

    def _extract_value_range(
        self,
        layer_name: str,
        property_name: Optional[str],
        layer_info: Dict,
    ) -> Tuple[float, float]:
        """Extract min/max values from layer data."""
        vmin, vmax = 0.0, 1.0

        # First, try to use cached values from the current rendering
        # if this is the currently rendered property
        if (self._current_property == property_name and
            self._vmin is not None and self._vmax is not None):
            logger.debug(f"Using cached vmin/vmax: {self._vmin}, {self._vmax}")
            return float(self._vmin), float(self._vmax)

        # Also check the layer_info for stored clim values
        current_vmin = layer_info.get('current_vmin')
        current_vmax = layer_info.get('current_vmax')
        if current_vmin is not None and current_vmax is not None:
            logger.debug(f"Using layer_info vmin/vmax: {current_vmin}, {current_vmax}")
            return float(current_vmin), float(current_vmax)

        # For Block Model, get data directly from renderer's current_model
        if self.renderer is not None and ('block' in layer_name.lower() or 'Block Model' in layer_name):
            try:
                current_model = getattr(self.renderer, 'current_model', None)
                if current_model is not None:
                    # Try to extract property name from layer name if not provided
                    prop_to_use = property_name
                    if not prop_to_use and ':' in layer_name:
                        # Extract property from "Block Model: PropertyName" format
                        prop_to_use = layer_name.split(':', 1)[1].strip()
                        logger.debug(f"Extracted property name '{prop_to_use}' from layer name '{layer_name}'")

                    if prop_to_use:
                        prop_data = current_model.get_property(prop_to_use)
                        if prop_data is not None:
                            finite_data = prop_data[np.isfinite(prop_data)]
                            if len(finite_data) > 0:
                                vmin = float(np.nanmin(finite_data))
                                vmax = float(np.nanmax(finite_data))
                                logger.info(f"Extracted vmin/vmax from block model property '{prop_to_use}': {vmin}, {vmax}")
                                return vmin, vmax
            except Exception as e:
                logger.debug(f"Failed to extract range from block model: {e}")

        # Try to get from mesh actor's scalar range
        if self.renderer is not None:
            try:
                mesh_actor = getattr(self.renderer, 'mesh_actor', None)
                if mesh_actor is not None:
                    mapper = mesh_actor.GetMapper()
                    if mapper is not None:
                        scalar_range = mapper.GetScalarRange()
                        if scalar_range and len(scalar_range) == 2:
                            vmin, vmax = float(scalar_range[0]), float(scalar_range[1])
                            logger.info(f"Extracted vmin/vmax from mesh_actor scalar range: {vmin}, {vmax}")
                            if vmin < vmax:
                                return vmin, vmax
            except Exception as e:
                logger.debug(f"Failed to get scalar range from mesh actor: {e}")

        # Fallback: try to extract from raw data
        data = layer_info.get('data')
        if data is not None and property_name:
            try:
                if hasattr(data, 'point_data'):
                    prop_data = data.point_data.get(property_name)
                    if prop_data is not None:
                        finite_data = prop_data[np.isfinite(prop_data)]
                        if len(finite_data) > 0:
                            vmin = float(np.nanmin(finite_data))
                            vmax = float(np.nanmax(finite_data))
                            logger.debug(f"Extracted vmin/vmax from point_data: {vmin}, {vmax}")
            except Exception:
                pass

        # Ensure valid range
        if vmin >= vmax:
            vmax = vmin + 1.0

        logger.debug(f"Final vmin/vmax for '{property_name}': {vmin}, {vmax}")
        return vmin, vmax

    def _extract_category_colors_from_layer(
        self,
        layer_info: Dict,
        categories: List,
    ) -> Dict[object, Tuple[float, float, float, float]]:
        """
        Extract actual rendered colors from layer data.

        For drillholes, colors are stored in data['lith_colors'].
        Colors can be hex strings ('#FF0000') or RGB tuples ((1.0, 0.0, 0.0)).
        """
        colors = {}
        data = layer_info.get('data')

        if data is None:
            logger.debug("No data in layer_info, cannot extract colors")
            return colors

        # For drillholes, extract from lith_colors
        if isinstance(data, dict) and 'lith_colors' in data:
            lith_colors = data.get('lith_colors', {})
            logger.info(f"Found {len(lith_colors)} lith_colors in layer data")

            for cat in categories:
                color_val = lith_colors.get(cat)
                if color_val is not None:
                    rgba = self._convert_color_to_rgba(color_val)
                    if rgba:
                        colors[cat] = rgba
                        logger.debug(f"Extracted color for '{cat}': {color_val} -> {rgba}")
                    else:
                        logger.warning(f"Failed to convert color for '{cat}': {color_val}")
                else:
                    logger.debug(f"No color found for category '{cat}' in lith_colors")

            logger.info(f"Extracted {len(colors)} category colors from lith_colors")
        else:
            logger.debug(f"Data is not dict or missing lith_colors: type={type(data)}, keys={data.keys() if isinstance(data, dict) else 'N/A'}")

        return colors

    def _convert_color_to_rgba(
        self,
        color_val: Any,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Convert various color formats to RGBA float tuple."""
        try:
            if isinstance(color_val, str):
                # Hex color string
                qc = QColor(color_val)
                if qc.isValid():
                    return (qc.redF(), qc.greenF(), qc.blueF(), 1.0)
            elif isinstance(color_val, (tuple, list, np.ndarray)):
                arr = np.asarray(color_val, dtype=float).flatten()
                if arr.size >= 3:
                    # Check if values are in 0-255 range or 0-1 range
                    if np.any(arr[:3] > 1.01):
                        arr = arr / 255.0
                    r, g, b = float(arr[0]), float(arr[1]), float(arr[2])
                    a = float(arr[3]) if arr.size > 3 else 1.0
                    return (r, g, b, a)
            elif isinstance(color_val, QColor):
                return (color_val.redF(), color_val.greenF(), color_val.blueF(), color_val.alphaF())
        except Exception as e:
            logger.debug(f"Failed to convert color {color_val}: {e}")

        return None

    def _generate_category_colors(
        self,
        categories: List,
    ) -> Dict[object, Tuple[float, float, float, float]]:
        """Generate colors for categories using a colormap."""
        colors = {}

        if not categories:
            return colors

        try:
            cmap = cm.get_cmap('tab20')
            for i, cat in enumerate(categories):
                n = i / max(1, len(categories) - 1) if len(categories) > 1 else 0.5
                rgba = cmap(n % 1.0)
                colors[cat] = tuple(float(x) for x in rgba)
        except Exception:
            # Fallback to gray
            for cat in categories:
                colors[cat] = (0.5, 0.5, 0.5, 1.0)

        return colors


def _to_qcolor(value: Union[str, Tuple[float, float, float], Tuple[int, int, int], QColor]) -> QColor:
    if isinstance(value, QColor):
        return QColor(value)
    try:
        if isinstance(value, str):
            return QColor(value)
        if isinstance(value, tuple) and len(value) == 3:
            if any(v > 1 for v in value):
                r, g, b = value
                return QColor(int(r), int(g), int(b))
            r, g, b = value
            return QColor(int(r * 255), int(g * 255), int(b * 255))
    except Exception:
        pass
    return QColor("#1f1f1f")


def _relative_luminance(color: QColor) -> float:
    r = color.redF()
    g = color.greenF()
    b = color.blueF()
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrasting_text_color(background: QColor) -> QColor:
    luminance = _relative_luminance(background)
    if luminance < 0.45:
        return QColor(245, 245, 245)
    return QColor(28, 32, 40)


class _LegendResizeFilter(QObject):
    """Keeps the legend anchored when the parent widget is resized."""

    def __init__(self, manager: LegendManager):
        super().__init__()
        self._manager_ref = weakref.ref(manager)

    def eventFilter(self, watched, event):  # type: ignore[override]
        try:
            if event.type() == QEvent.Type.Resize:
                manager = self._manager_ref()
                if manager is not None:
                    manager._ensure_position()
        except Exception:
            pass
        return False
