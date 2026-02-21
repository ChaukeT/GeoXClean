"""
File menu construction for GeoX.

Extracted from MainWindow to improve maintainability.
Handles file operations, project management, and exports.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
    ICONS_AVAILABLE = True
except ImportError:
    ICONS_AVAILABLE = False
    def get_menu_icon(category, name):
        return None


def build_file_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the File menu."""
    file_menu = menubar.addMenu("&File")
    
    # Project actions
    new_project_action = QAction(get_menu_icon("file", "new_project"), "&New Project", main_window)
    new_project_action.setStatusTip("Start a new, empty project")
    new_project_action.triggered.connect(main_window._new_project)
    file_menu.addAction(new_project_action)

    open_project_action = QAction(get_menu_icon("file", "open_project"), "Open &Project...", main_window)
    open_project_action.setStatusTip("Open a Block Model Viewer project file (.bmvproj)")
    open_project_action.triggered.connect(main_window.open_project)
    file_menu.addAction(open_project_action)

    save_project_action = QAction(get_menu_icon("file", "save_project"), "&Save Project", main_window)
    save_project_action.setShortcut(QKeySequence.StandardKey.Save)
    save_project_action.setStatusTip("Save current project")
    save_project_action.triggered.connect(main_window.save_project)
    file_menu.addAction(save_project_action)

    save_project_as_action = QAction(get_menu_icon("file", "save_as"), "Save Project &As...", main_window)
    save_project_as_action.setStatusTip("Save current project to a new file")
    save_project_as_action.triggered.connect(main_window.save_project_as)
    file_menu.addAction(save_project_as_action)

    file_menu.addSeparator()

    # File operations
    open_action = QAction(get_menu_icon("file", "open_file"), "&Open File...", main_window)
    open_action.setShortcut(QKeySequence.StandardKey.Open)
    open_action.setStatusTip("Open a block model file")
    open_action.triggered.connect(main_window.open_file)
    file_menu.addAction(open_action)
    
    block_model_load_action = QAction(get_menu_icon("file", "block_model_loading"), "Block Model Loading...", main_window)
    block_model_load_action.setStatusTip("Load block model CSV with auto-detect and manual column mapping")
    block_model_load_action.triggered.connect(main_window.open_block_model_import_panel)
    file_menu.addAction(block_model_load_action)
    
    # Recent Files submenu
    main_window.recent_files_menu = file_menu.addMenu("Recent Files")
    main_window._update_recent_files_menu()
    
    file_menu.addSeparator()

    # Session restore toggle
    main_window.restore_session_action = QAction("Restore Last Session on Startup", main_window)
    main_window.restore_session_action.setCheckable(True)
    try:
        from PyQt6.QtCore import QSettings
        s = QSettings("GeoX", "Session")
        restore_flag = s.value("restore_on_startup", True, type=bool)
        main_window.restore_session_action.setChecked(bool(restore_flag))
    except Exception:
        main_window.restore_session_action.setChecked(True)
    
    def _toggle_restore_session(checked: bool):
        try:
            from PyQt6.QtCore import QSettings
            s = QSettings("GeoX", "Session")
            s.setValue("restore_on_startup", bool(checked))
        except Exception:
            pass
    main_window.restore_session_action.triggered.connect(_toggle_restore_session)
    file_menu.addAction(main_window.restore_session_action)
    
    # Export actions
    screenshot_action = QAction(get_menu_icon("file", "screenshot"), "Export &Screenshot...", main_window)
    screenshot_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    screenshot_action.setStatusTip("Save current view as image (simple)")
    screenshot_action.triggered.connect(main_window.export_screenshot)
    file_menu.addAction(screenshot_action)

    advanced_screenshot_action = QAction(get_menu_icon("file", "advanced_screenshot"), "Advanced Screenshot Export...", main_window)
    advanced_screenshot_action.setStatusTip("Export branded screenshot with title, legend, and custom layout")
    advanced_screenshot_action.triggered.connect(main_window.export_advanced_screenshot)
    file_menu.addAction(advanced_screenshot_action)

    export_data_action = QAction(get_menu_icon("file", "export_data"), "Export &Data...", main_window)
    export_data_action.setStatusTip("Export block models, drillholes, or estimation results to various formats")
    export_data_action.triggered.connect(main_window.export_filtered_data)
    file_menu.addAction(export_data_action)

    export_model_action = QAction(get_menu_icon("file", "export_model"), "Export &Model (STL/OBJ)...", main_window)
    export_model_action.setStatusTip("Export 3D model to file")
    export_model_action.triggered.connect(main_window.export_model)
    file_menu.addAction(export_model_action)

    file_menu.addSeparator()
    
    # Application control
    restart_action = QAction(get_menu_icon("file", "restart"), "&Restart", main_window)
    restart_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
    restart_action.setStatusTip("Close and restart the application")
    restart_action.triggered.connect(main_window.restart_application)
    file_menu.addAction(restart_action)

    file_menu.addSeparator()
    
    # Scene control
    clear_scene_action = QAction(get_menu_icon("file", "clear_scene"), "&Clear Scene", main_window)
    clear_scene_action.setShortcut(QKeySequence("Ctrl+W"))
    clear_scene_action.setStatusTip("Remove block model and clear the scene")
    clear_scene_action.triggered.connect(main_window.clear_scene)
    file_menu.addAction(clear_scene_action)

    file_menu.addSeparator()
    
    # Exit
    exit_action = QAction(get_menu_icon("file", "exit"), "E&xit", main_window)
    exit_action.setShortcut(QKeySequence.StandardKey.Quit)
    exit_action.setStatusTip("Exit application")
    exit_action.triggered.connect(main_window.close)
    file_menu.addAction(exit_action)

    return file_menu

