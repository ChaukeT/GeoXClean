"""
View menu construction for GeoX.
"""

import logging
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence, QActionGroup

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None

logger = logging.getLogger(__name__)


def _safe(action, mw, method_name, *args):
    """Connect action.triggered to mw.method if it exists, else disable."""
    handler = getattr(mw, method_name, None)
    if handler is not None:
        if args:
            action.triggered.connect(lambda checked, h=handler, a=args: h(*a))
        else:
            action.triggered.connect(handler)
    else:
        action.setEnabled(False)
        logger.debug("view_menu: MainWindow.%s not found", method_name)


def build_view_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the View menu."""
    view_menu = menubar.addMenu("&View")

    reset_view_action = QAction(get_menu_icon("view", "reset_view"), "&Reset View", main_window)
    reset_view_action.setShortcut(QKeySequence("R"))
    reset_view_action.setStatusTip("Reset camera to default position")
    _safe(reset_view_action, main_window, 'reset_camera')
    view_menu.addAction(reset_view_action)

    view_menu.addSeparator()

    # View presets submenu
    view_presets_menu = view_menu.addMenu("View &Presets")
    presets = [
        ("&Top", "1", "Top"), ("&Bottom", "2", "Bottom"),
        ("&Front", "3", "Front"), ("&Back", "4", "Back"),
        ("&Right", "5", "Right"), ("&Left", "6", "Left"),
        ("&Isometric", "7", "Isometric"),
    ]
    for name, shortcut, preset in presets:
        action = QAction(name, main_window)
        action.setShortcut(QKeySequence(shortcut))
        action.setStatusTip(f"Set {preset.lower()} view")
        if hasattr(main_window, 'set_view_preset'):
            action.triggered.connect(lambda checked, p=preset: main_window.set_view_preset(p))
        view_presets_menu.addAction(action)

    view_menu.addSeparator()

    # Orthographic toggle
    proj_action = QAction(get_menu_icon("view", "orthographic"), "&Orthographic Projection", main_window)
    proj_action.setShortcut(QKeySequence("O"))
    proj_action.setCheckable(True)
    proj_action.setStatusTip("Toggle orthographic/perspective projection")
    _safe(proj_action, main_window, 'toggle_projection')
    main_window.projection_action = proj_action
    view_menu.addAction(proj_action)

    view_menu.addSeparator()

    # Data Registry Status
    registry_status_action = QAction(get_menu_icon("view", "data_registry"), "Data Registry &Status...", main_window)
    registry_status_action.setStatusTip("View DataRegistry status and data flow")
    _safe(registry_status_action, main_window, 'open_data_registry_status_panel')
    view_menu.addAction(registry_status_action)

    # Registry block models submenu
    reg_bm_menu = view_menu.addMenu("Registry Block &Models")
    reg_bm_menu.setToolTipsVisible(True)
    main_window.registry_block_models_menu = reg_bm_menu
    if hasattr(main_window, 'populate_registry_block_models_menu'):
        reg_bm_menu.aboutToShow.connect(main_window.populate_registry_block_models_menu)

    view_menu.addSeparator()

    # Apply view defaults
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings("GeoX", "View")
        preset = str(s.value("default_lighting", "balanced"))
        if hasattr(main_window, 'apply_lighting_preset'):
            main_window.apply_lighting_preset(preset)
    except Exception:
        pass

    view_menu.addSeparator()

    # ── Panel visibility toggles ──────────────────────────────
    panels_submenu = view_menu.addMenu("&Panels")

    def _dock_toggle(dock_attr, label):
        a = QAction(label, main_window)
        a.setCheckable(True)
        a.setChecked(True)
        dock = getattr(main_window, dock_attr, None)
        if dock is not None and hasattr(main_window, '_toggle_dock'):
            a.triggered.connect(lambda checked, d=dock: main_window._toggle_dock(d, checked))
        else:
            a.setEnabled(False)
        return a

    main_window.controls_scene_action = _dock_toggle('left_dock', "&Controls && Scene")
    panels_submenu.addAction(main_window.controls_scene_action)

    main_window.gc_decision_action = _dock_toggle('gc_decision_dock', "&GC Decision Engine")
    panels_submenu.addAction(main_window.gc_decision_action)

    main_window.drillhole_explorer_action = _dock_toggle('drillhole_control_dock', "&Drillhole Explorer")
    panels_submenu.addAction(main_window.drillhole_explorer_action)

    panels_submenu.addSeparator()

    multi_legend_action = QAction("&Multi-Legend Panel", main_window)
    multi_legend_action.setShortcut(QKeySequence("Ctrl+L"))
    multi_legend_action.setCheckable(True)
    multi_legend_action.setChecked(False)
    _safe(multi_legend_action, main_window, '_toggle_multi_legend')
    main_window.multi_legend_action = multi_legend_action
    panels_submenu.addAction(multi_legend_action)

    classic_legend_action = QAction("Classic &Legend", main_window)
    classic_legend_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
    classic_legend_action.setCheckable(True)
    classic_legend_action.setChecked(False)
    _safe(classic_legend_action, main_window, '_toggle_classic_legend')
    main_window.classic_legend_action = classic_legend_action
    panels_submenu.addAction(classic_legend_action)

    if hasattr(main_window, '_update_all_dock_menu_states'):
        try:
            main_window._update_all_dock_menu_states()
        except Exception:
            pass

    view_menu.addSeparator()

    # Cross-Section Tool (under development)
    cs_action = QAction(get_menu_icon("view", "cross_section"), "Cross-&Section Tool", main_window)
    cs_action.setStatusTip("Cross-section tool (under development)")
    cs_action.triggered.connect(
        lambda: main_window.status_bar.showMessage("Cross-Section Tool is under development", 3000)
    )
    view_menu.addAction(cs_action)

    view_menu.addSeparator()

    # View block model data
    vd_action = QAction(get_menu_icon("view", "view_data"), "View Block Model &Data", main_window)
    vd_action.setShortcut(QKeySequence("Ctrl+D"))
    vd_action.setStatusTip("View block model data in table format")
    _safe(vd_action, main_window, 'open_data_viewer_window')
    vd_action.setEnabled(getattr(main_window, 'current_model', None) is not None)
    main_window.view_data_action = vd_action
    view_menu.addAction(vd_action)

    # View drillhole data
    vdd_action = QAction(get_menu_icon("view", "drillhole_data"), "View &Drillhole Data", main_window)
    vdd_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
    vdd_action.setStatusTip("View drillhole data in table format")
    _safe(vdd_action, main_window, 'open_drillhole_data_viewer_window')
    view_menu.addAction(vdd_action)

    view_menu.addSeparator()

    # Lighting presets
    lighting_menu = view_menu.addMenu("&Lighting Presets")
    for label, preset in [("&Soft", "soft"), ("&Balanced", "balanced"), ("S&harp", "sharp")]:
        a = QAction(label, main_window)
        if hasattr(main_window, 'apply_lighting_preset'):
            a.triggered.connect(lambda checked, p=preset: main_window.apply_lighting_preset(p))
        lighting_menu.addAction(a)

    view_menu.addSeparator()

    # Theme submenu
    theme_menu = view_menu.addMenu("&Theme")
    light_theme_action = QAction("&Light", main_window)
    light_theme_action.setCheckable(True)
    if hasattr(main_window, 'set_theme'):
        light_theme_action.triggered.connect(lambda: main_window.set_theme("light"))
    theme_menu.addAction(light_theme_action)

    dark_theme_action = QAction("&Dark", main_window)
    dark_theme_action.setCheckable(True)
    if hasattr(main_window, 'set_theme'):
        dark_theme_action.triggered.connect(lambda: main_window.set_theme("dark"))
    theme_menu.addAction(dark_theme_action)

    main_window.theme_action_group = QActionGroup(main_window)
    main_window.theme_action_group.setExclusive(True)
    main_window.theme_action_group.addAction(light_theme_action)
    main_window.theme_action_group.addAction(dark_theme_action)
    main_window.light_theme_action = light_theme_action
    main_window.dark_theme_action = dark_theme_action

    view_menu.addSeparator()

    # Workspace layout
    workspace_menu = view_menu.addMenu("&Workspace Layout")
    for label, layout_id in [("Resource Evaluation", "resource"),
                              ("Planning && Design", "planning"),
                              ("Uncertainty && Analytics", "analytics")]:
        a = QAction(label, main_window)
        if hasattr(main_window, 'load_workspace_layout'):
            a.triggered.connect(lambda checked, lid=layout_id: main_window.load_workspace_layout(lid))
        else:
            a.setEnabled(False)
        workspace_menu.addAction(a)

    workspace_menu.addSeparator()

    reset_ws = QAction("Reset Workspace", main_window)
    _safe(reset_ws, main_window, 'reset_workspace_layout')
    workspace_menu.addAction(reset_ws)

    save_ws = QAction("Save Workspace Layout...", main_window)
    _safe(save_ws, main_window, 'save_workspace_layout')
    workspace_menu.addAction(save_ws)

    load_ws = QAction("Load Workspace Layout...", main_window)
    _safe(load_ws, main_window, 'load_workspace_layout_file')
    workspace_menu.addAction(load_ws)

    return view_menu
