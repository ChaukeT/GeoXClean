"""
LegendController: Mediates between renderer, legend manager, and legend widget.
Handles legend state updates, persistence, and unified event flow.
"""
from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import QObject
import json
from pathlib import Path

from .legend_logging import get_legend_logger

if TYPE_CHECKING:
    from .multi_legend_widget import MultiLegendWidget

logger = get_legend_logger("controller")


class LegendController(QObject):
    def __init__(self, legend_widget, legend_manager, config=None, multi_legend_widget: Optional["MultiLegendWidget"] = None):
        super().__init__()
        self.legend_widget = legend_widget
        self.legend_manager = legend_manager
        self.multi_legend_widget = multi_legend_widget
        self.config = config or {}
        self._state_path = Path.home() / ".block_model_viewer" / "legend_state.json"
        self._multi_state_path = Path.home() / ".block_model_viewer" / "multi_legend_state.json"
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("LegendController initialised (state_path=%s)", self._state_path)
        self._connect_signals()
        self._restore_state()

    def _connect_signals(self):
        # Widget signals
        self.legend_widget.size_changed.connect(self._on_size_changed)
        self.legend_widget.floating_position_changed.connect(self._on_position_changed)
        self.legend_widget.background_color_changed.connect(self._on_background_changed)
        self.legend_widget.dock_requested.connect(self._on_dock_requested)
        self.legend_widget.colormap_changed.connect(self._on_colormap_changed)
        self.legend_widget.reverse_toggled.connect(self._on_reverse_toggled)
        self.legend_widget.orientation_changed.connect(self._on_orientation_changed)
        self.legend_widget.value_range_changed.connect(self._on_value_range_changed)
        self.legend_widget.category_toggled.connect(self._on_category_toggled)

        # Multi-legend widget signals
        if self.multi_legend_widget is not None:
            self.multi_legend_widget.config_changed.connect(self._on_multi_config_changed)
        logger.debug("LegendController signals connected")

    def bind_multi_legend_widget(self, multi_widget: "MultiLegendWidget") -> None:
        """Bind a multi-legend widget for persistence."""
        self.multi_legend_widget = multi_widget
        multi_widget.config_changed.connect(self._on_multi_config_changed)
        self._restore_multi_state()

    def _on_multi_config_changed(self, config_dict: dict) -> None:
        """Handle multi-legend config changes."""
        logger.debug("Persisting multi-legend config with %d elements", len(config_dict.get('elements', [])))
        self._persist_multi_state(config_dict)

    def _persist_multi_state(self, config_dict: dict) -> None:
        """Save multi-legend state to file."""
        try:
            with self._multi_state_path.open('w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logger.debug("Multi-legend state persisted to %s", self._multi_state_path)
        except Exception as exc:
            logger.exception("Failed to persist multi-legend state: %s", exc)

    def _restore_multi_state(self) -> None:
        """Restore multi-legend state from file."""
        if self.multi_legend_widget is None:
            return

        try:
            if not self._multi_state_path.exists():
                logger.debug("No multi-legend state file found at %s", self._multi_state_path)
                return

            with self._multi_state_path.open('r', encoding='utf-8') as f:
                config_dict = json.load(f)

            logger.debug("Restoring multi-legend state with %d elements", len(config_dict.get('elements', [])))
            self.multi_legend_widget.load_config(config_dict)
        except Exception as exc:
            logger.exception("Failed to restore multi-legend state: %s", exc)

    def _on_size_changed(self, w, h):
        self.config['size'] = (w, h)
        logger.debug("Persisting legend size change -> %s", self.config['size'])
        self._persist_state()

    def _on_position_changed(self, x, y):
        self.config['position'] = (x, y)
        logger.debug("Persisting legend position change -> %s", self.config['position'])
        self._persist_state()

    def _on_background_changed(self, color):
        self.config['background_color'] = color.name()
        logger.debug("Persisting legend background -> %s", self.config['background_color'])
        self._persist_state()

    def _on_dock_requested(self, anchor):
        self.config['anchor'] = anchor
        logger.debug("Persisting legend anchor -> %s", anchor)
        self._persist_state()

    def _on_colormap_changed(self, cmap):
        self.config['cmap_name'] = cmap
        logger.debug("Persisting legend colormap -> %s", cmap)
        self._persist_state()

    def _on_reverse_toggled(self, reverse):
        self.config['reverse'] = reverse
        logger.debug("Persisting legend reverse -> %s", reverse)
        self._persist_state()

    def _on_orientation_changed(self, orientation):
        self.config['orientation'] = orientation
        logger.debug("Persisting legend orientation -> %s", orientation)
        self._persist_state()

    def _on_value_range_changed(self, vmin, vmax):
        self.config['vmin'] = vmin
        self.config['vmax'] = vmax
        logger.debug("Persisting legend range -> (%s, %s)", vmin, vmax)
        self._persist_state()

    def _on_category_toggled(self, category, visible):
        if 'category_visible' not in self.config:
            self.config['category_visible'] = {}
        self.config['category_visible'][str(category)] = visible
        logger.debug("Persisting legend category visibility %s -> %s", category, visible)
        self._persist_state()

    def _persist_state(self):
        # Save to file or session (example: JSON file)
        try:
            with self._state_path.open('w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            logger.debug("Legend state persisted to %s", self._state_path)
        except Exception as exc:
            logger.exception("Failed to persist legend state: %s", exc)

    def _restore_state(self):
        # Load from file or session
        try:
            if not self._state_path.exists():
                logger.debug("No legend state file found at %s", self._state_path)
                return
            with self._state_path.open('r', encoding='utf-8') as f:
                state = json.load(f)
            logger.debug("Restoring legend state from %s", self._state_path)
            self.apply_state(state)
        except Exception as exc:
            logger.exception("Failed to restore legend state: %s", exc)

    def apply_state(self, state):
        # Apply persisted state to widget
        logger.debug("Applying legend state: %s", state)
        if 'size' in state:
            self.legend_widget.resize(*state['size'])
        if 'position' in state:
            self.legend_widget.move(*state['position'])
        if 'background_color' in state:
            from PyQt6.QtGui import QColor
            self.legend_widget.config.background_color = QColor(state['background_color'])
        if 'anchor' in state:
            self.legend_widget.set_current_anchor(state['anchor'])
        if 'cmap_name' in state:
            self.legend_widget.config.cmap_name = state['cmap_name']
        if 'reverse' in state:
            self.legend_widget.config.reverse = state['reverse']
        if 'orientation' in state:
            self.legend_widget._orientation = state['orientation']
        if 'vmin' in state and 'vmax' in state:
            self.legend_widget.config.vmin = state['vmin']
            self.legend_widget.config.vmax = state['vmax']
        if 'category_visible' in state:
            self.legend_widget._category_visible = state['category_visible']
        self.legend_widget.update()
