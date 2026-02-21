"""
View menu construction for GeoX.

Extracted from MainWindow to improve maintainability.
Handles view presets, panels, lighting, themes, and workspace layouts.
"""

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


def build_view_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the View menu."""
    view_menu = menubar.addMenu("&View")
    
    reset_view_action = QAction(get_menu_icon("view", "reset_view"), "&Reset View", main_window)
    reset_view_action.setShortcut(QKeySequence("R"))
    reset_view_action.setStatusTip("Reset camera to default position")
    reset_view_action.triggered.connect(main_window.reset_camera)
    view_menu.addAction(reset_view_action)
    
    view_menu.addSeparator()
    
    # View presets submenu
    view_presets_menu = view_menu.addMenu("View &Presets")
    
    presets = [
        ("&Top", "1", "Top"),
        ("&Bottom", "2", "Bottom"),
        ("&Front", "3", "Front"),
        ("&Back", "4", "Back"),
        ("&Right", "5", "Right"),
        ("&Left", "6", "Left"),
        ("&Isometric", "7", "Isometric")
    ]
    
    for name, shortcut, preset in presets:
        action = QAction(name, main_window)
        action.setShortcut(QKeySequence(shortcut))
        action.setStatusTip(f"Set {preset.lower()} view")
        action.triggered.connect(lambda checked, p=preset: main_window.set_view_preset(p))
        view_presets_menu.addAction(action)
    
    view_menu.addSeparator()
    
    # Toggle actions
    main_window.projection_action = QAction(get_menu_icon("view", "orthographic"), "&Orthographic Projection", main_window)
    main_window.projection_action.setShortcut(QKeySequence("O"))
    main_window.projection_action.setCheckable(True)
    main_window.projection_action.setStatusTip("Toggle orthographic/perspective projection")
    main_window.projection_action.triggered.connect(main_window.toggle_projection)
    view_menu.addAction(main_window.projection_action)
    
    view_menu.addSeparator()
    
    # Data Registry Status
    registry_status_action = QAction(get_menu_icon("view", "data_registry"), "Data Registry &Status...", main_window)
    registry_status_action.setStatusTip("View DataRegistry status and data flow")
    registry_status_action.triggered.connect(main_window.open_data_registry_status_panel)
    view_menu.addAction(registry_status_action)

    # Registry block models (dynamic submenu populated on open)
    main_window.registry_block_models_menu = view_menu.addMenu("Registry Block &Models")
    main_window.registry_block_models_menu.setToolTipsVisible(True)
    main_window.registry_block_models_menu.aboutToShow.connect(main_window.populate_registry_block_models_menu)

    view_menu.addSeparator()
    
    # Apply view defaults from Preferences
    try:
        from PyQt6.QtCore import QSettings
        s_view_defaults = QSettings("GeoX", "View")
        lighting_default = str(s_view_defaults.value("default_lighting", "balanced"))
        if hasattr(main_window, 'apply_lighting_preset'):
            main_window.apply_lighting_preset(lighting_default)
    except Exception:
        pass
    
    view_menu.addSeparator()
    
    # Panel visibility toggles
    panels_submenu = view_menu.addMenu("&Panels")
    
    # Controls & Scene (left dock with Property Controls, Scene Inspector)
    main_window.controls_scene_action = QAction("&Controls & Scene", main_window)
    main_window.controls_scene_action.setCheckable(True)
    main_window.controls_scene_action.setChecked(True)
    main_window.controls_scene_action.setStatusTip("Toggle Controls & Scene panel (Property Controls, Scene Inspector)")
    main_window.controls_scene_action.triggered.connect(lambda checked: main_window._toggle_dock(main_window.left_dock, checked))
    panels_submenu.addAction(main_window.controls_scene_action)
    
    # GC Decision Engine
    main_window.gc_decision_action = QAction("&GC Decision Engine", main_window)
    main_window.gc_decision_action.setCheckable(True)
    main_window.gc_decision_action.setChecked(True)
    main_window.gc_decision_action.setStatusTip("Toggle GC Decision Engine panel")
    main_window.gc_decision_action.triggered.connect(lambda checked: main_window._toggle_dock(main_window.gc_decision_dock, checked))
    panels_submenu.addAction(main_window.gc_decision_action)
    
    # Drillhole Explorer
    main_window.drillhole_explorer_action = QAction("&Drillhole Explorer", main_window)
    main_window.drillhole_explorer_action.setCheckable(True)
    main_window.drillhole_explorer_action.setChecked(True)
    main_window.drillhole_explorer_action.setStatusTip("Toggle Drillhole Explorer panel")
    main_window.drillhole_explorer_action.triggered.connect(lambda checked: main_window._toggle_dock(main_window.drillhole_control_dock, checked))
    panels_submenu.addAction(main_window.drillhole_explorer_action)

    panels_submenu.addSeparator()

    # Multi-Legend Panel (interactive legend with add/remove)
    main_window.multi_legend_action = QAction("&Multi-Legend Panel", main_window)
    main_window.multi_legend_action.setShortcut(QKeySequence("Ctrl+L"))
    main_window.multi_legend_action.setCheckable(True)
    main_window.multi_legend_action.setChecked(False)
    main_window.multi_legend_action.setStatusTip("Toggle interactive multi-element legend panel")
    main_window.multi_legend_action.triggered.connect(main_window._toggle_multi_legend)
    panels_submenu.addAction(main_window.multi_legend_action)

    # Classic Legend (continuous/discrete colorbar legend)
    main_window.classic_legend_action = QAction("Classic &Legend", main_window)
    main_window.classic_legend_action.setShortcut(QKeySequence("Ctrl+Shift+L"))
    main_window.classic_legend_action.setCheckable(True)
    main_window.classic_legend_action.setChecked(False)
    main_window.classic_legend_action.setStatusTip("Toggle classic colorbar/discrete legend (switches from multi-legend mode)")
    main_window.classic_legend_action.triggered.connect(main_window._toggle_classic_legend)
    panels_submenu.addAction(main_window.classic_legend_action)

    # Update initial checked states based on current dock visibility
    main_window._update_all_dock_menu_states()
    
    view_menu.addSeparator()
    
    # Cross-Section Tool
    cross_section_view_action = QAction(get_menu_icon("view", "cross_section"), "Cross-&Section Tool", main_window)
    cross_section_view_action.setStatusTip("Slice 3D data with a plane (N-S, E-W, or custom) - works with block models and drillholes")
    cross_section_view_action.triggered.connect(main_window.open_cross_section_panel)
    view_menu.addAction(cross_section_view_action)

    view_menu.addSeparator()
    
    # View Block Model Data action
    main_window.view_data_action = QAction(get_menu_icon("view", "view_data"), "View Block Model &Data", main_window)
    main_window.view_data_action.setShortcut(QKeySequence("Ctrl+D"))
    main_window.view_data_action.setStatusTip("View block model data in table format")
    main_window.view_data_action.triggered.connect(main_window.open_data_viewer_window)
    # Disabled until a block model is loaded
    try:
        main_window.view_data_action.setEnabled(main_window.current_model is not None)
    except Exception:
        main_window.view_data_action.setEnabled(False)
    view_menu.addAction(main_window.view_data_action)

    # View Drillhole Data action
    view_drill_data_action = QAction(get_menu_icon("view", "drillhole_data"), "View &Drillhole Data", main_window)
    view_drill_data_action.setShortcut(QKeySequence("Ctrl+Shift+D"))
    view_drill_data_action.setStatusTip("View drillhole data in table format")
    view_drill_data_action.triggered.connect(main_window.open_drillhole_data_viewer_window)
    view_menu.addAction(view_drill_data_action)
    
    view_menu.addSeparator()
    
    # Lighting presets submenu
    lighting_menu = view_menu.addMenu("&Lighting Presets")
    
    soft_light = QAction("&Soft", main_window)
    soft_light.triggered.connect(lambda: main_window.apply_lighting_preset("soft"))
    lighting_menu.addAction(soft_light)
    
    balanced_light = QAction("&Balanced", main_window)
    balanced_light.triggered.connect(lambda: main_window.apply_lighting_preset("balanced"))
    lighting_menu.addAction(balanced_light)
    
    sharp_light = QAction("S&harp", main_window)
    sharp_light.triggered.connect(lambda: main_window.apply_lighting_preset("sharp"))
    lighting_menu.addAction(sharp_light)
    
    view_menu.addSeparator()
    
    # Theme submenu
    theme_menu = view_menu.addMenu("&Theme")
    
    light_theme_action = QAction("&Light", main_window)
    light_theme_action.setCheckable(True)
    light_theme_action.triggered.connect(lambda: main_window.set_theme("light"))
    theme_menu.addAction(light_theme_action)
    
    dark_theme_action = QAction("&Dark", main_window)
    dark_theme_action.setCheckable(True)
    dark_theme_action.triggered.connect(lambda: main_window.set_theme("dark"))
    theme_menu.addAction(dark_theme_action)
    
    # Create action group for exclusive selection
    main_window.theme_action_group = QActionGroup(main_window)
    main_window.theme_action_group.setExclusive(True)
    main_window.theme_action_group.addAction(light_theme_action)
    main_window.theme_action_group.addAction(dark_theme_action)
    
    # Store theme actions for later updates
    main_window.light_theme_action = light_theme_action
    main_window.dark_theme_action = dark_theme_action
    
    view_menu.addSeparator()
    
    # Workspace Layout submenu
    workspace_menu = view_menu.addMenu("&Workspace Layout")
    
    resource_layout = QAction("Resource Evaluation", main_window)
    resource_layout.triggered.connect(lambda: main_window.load_workspace_layout("resource"))
    workspace_menu.addAction(resource_layout)
    
    planning_layout = QAction("Planning & Design", main_window)
    planning_layout.triggered.connect(lambda: main_window.load_workspace_layout("planning"))
    workspace_menu.addAction(planning_layout)
    
    analytics_layout = QAction("Uncertainty & Analytics", main_window)
    analytics_layout.triggered.connect(lambda: main_window.load_workspace_layout("analytics"))
    workspace_menu.addAction(analytics_layout)
    
    workspace_menu.addSeparator()
    
    reset_workspace_action = QAction("Reset Workspace", main_window)
    reset_workspace_action.triggered.connect(main_window.reset_workspace_layout)
    workspace_menu.addAction(reset_workspace_action)
    
    save_workspace_action = QAction("Save Workspace Layout...", main_window)
    save_workspace_action.triggered.connect(main_window.save_workspace_layout)
    workspace_menu.addAction(save_workspace_action)
    
    load_workspace_action = QAction("Load Workspace Layout...", main_window)
    load_workspace_action.triggered.connect(main_window.load_workspace_layout_file)
    workspace_menu.addAction(load_workspace_action)
    
    return view_menu

