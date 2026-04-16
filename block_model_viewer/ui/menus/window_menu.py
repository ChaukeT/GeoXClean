"""
Window menu — GeoX consolidated menu system.

Position: 10th (second-to-last, universal convention).
Handles: dock panel toggles, legends.

Only exposes true dock/utility panels — small panels that sit alongside
the viewport and are toggled on/off. Full tool panels (Kriging, etc.)
are opened ONLY from their domain menus.
"""

import logging
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


# Dock-only panels: (panel_id, display_name, tooltip)
DOCK_PANELS = [
    ("DrillholeInfoPanel", "&Drillhole Info", "Drillhole information display"),
    ("DisplaySettingsPanel", "&Display Settings", "Rendering and display settings"),
]


def build_window_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    menu = menubar.addMenu("&Window")

    # ── Permanent Dock Toggles ───────────────────────────────────
    main_window.controls_scene_action = QAction("&Property Controls", main_window)
    main_window.controls_scene_action.setCheckable(True)
    main_window.controls_scene_action.setChecked(True)
    main_window.controls_scene_action.setStatusTip("Toggle Property Controls panel (left dock)")
    main_window.controls_scene_action.triggered.connect(
        lambda checked: main_window._toggle_dock(main_window.left_dock, checked)
    )
    menu.addAction(main_window.controls_scene_action)

    main_window.gc_decision_action = QAction("&GC Decision Engine", main_window)
    main_window.gc_decision_action.setCheckable(True)
    main_window.gc_decision_action.setChecked(True)
    main_window.gc_decision_action.setStatusTip("Toggle GC Decision Engine panel")
    main_window.gc_decision_action.triggered.connect(
        lambda checked: main_window._toggle_dock(main_window.gc_decision_dock, checked)
    )
    menu.addAction(main_window.gc_decision_action)

    main_window.drillhole_explorer_action = QAction("&Drillhole Explorer", main_window)
    main_window.drillhole_explorer_action.setCheckable(True)
    main_window.drillhole_explorer_action.setChecked(True)
    main_window.drillhole_explorer_action.setStatusTip("Toggle Drillhole Explorer panel")
    main_window.drillhole_explorer_action.triggered.connect(
        lambda checked: main_window._toggle_dock(main_window.drillhole_control_dock, checked)
    )
    menu.addAction(main_window.drillhole_explorer_action)

    main_window._update_all_dock_menu_states()

    menu.addSeparator()

    # ── Utility Dock Panels ──────────────────────────────────────
    _add_dock_panels(main_window, menu)

    menu.addSeparator()

    # ── Legends ──────────────────────────────────────────────────
    legends_menu = menu.addMenu("&Legends")

    main_window.multi_legend_action = QAction("&Multi-Legend Panel", main_window)
    main_window.multi_legend_action.setShortcut("Ctrl+L")
    main_window.multi_legend_action.setCheckable(True)
    main_window.multi_legend_action.setChecked(False)
    main_window.multi_legend_action.setStatusTip("Toggle interactive multi-element legend")
    main_window.multi_legend_action.triggered.connect(main_window._toggle_multi_legend)
    legends_menu.addAction(main_window.multi_legend_action)

    main_window.classic_legend_action = QAction("Classic &Legend", main_window)
    main_window.classic_legend_action.setShortcut("Ctrl+Alt+L")
    main_window.classic_legend_action.setCheckable(True)
    main_window.classic_legend_action.setChecked(False)
    main_window.classic_legend_action.setStatusTip("Toggle classic colorbar/discrete legend")
    main_window.classic_legend_action.triggered.connect(main_window._toggle_classic_legend)
    legends_menu.addAction(main_window.classic_legend_action)

    return menu


# ── Helpers ──────────────────────────────────────────────────────

def _add_dock_panels(main_window, parent_menu):
    """Add checkable toggle actions for dock-only panels."""
    pm = getattr(main_window, 'panel_manager', None)
    if not pm:
        return

    for panel_id, display_name, tooltip in DOCK_PANELS:
        try:
            info = pm.get_panel_info(panel_id)
            if info is None:
                continue
        except Exception:
            continue

        action = QAction(display_name, main_window)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        try:
            action.setChecked(pm.is_panel_visible(panel_id))
        except Exception:
            action.setChecked(False)

        pid = panel_id
        action.triggered.connect(lambda checked, p=pid: _toggle_panel(pm, p))
        if info is not None:
            info.menu_action = action
        parent_menu.addAction(action)


def _toggle_panel(panel_manager, panel_id: str):
    try:
        panel_manager.toggle_panel(panel_id)
    except Exception as e:
        logger.error(f"Error toggling panel {panel_id}: {e}")
