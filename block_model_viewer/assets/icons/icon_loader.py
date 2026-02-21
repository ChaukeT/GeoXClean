"""
GeoX Icon Loader Utility

Provides convenient functions to load SVG icons for menus and toolbars.
Icons are organized by category under assets/icons/menu/.

Usage:
    from block_model_viewer.assets.icons.icon_loader import get_icon, get_menu_icon
    
    # Load a specific icon by category and name
    icon = get_menu_icon("file", "save_project")
    action.setIcon(icon)
    
    # Load icon by full path
    icon = get_icon("menu/file/save_project.svg")
    action.setIcon(icon)
"""

import os
from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import QSize

# Base path for icons
ICONS_BASE_PATH = Path(__file__).parent


def get_icon(relative_path: str) -> QIcon:
    """
    Load an icon from the icons directory.
    
    Args:
        relative_path: Path relative to icons directory (e.g., "menu/file/save_project.svg")
        
    Returns:
        QIcon: The loaded icon, or empty icon if not found
    """
    icon_path = ICONS_BASE_PATH / relative_path
    if icon_path.exists():
        return QIcon(str(icon_path))
    return QIcon()


def get_menu_icon(category: str, name: str) -> QIcon:
    """
    Load a menu icon by category and name.
    
    Args:
        category: Menu category (e.g., "file", "edit", "view", "tools")
        name: Icon name without extension (e.g., "save_project", "undo")
        
    Returns:
        QIcon: The loaded icon, or empty icon if not found
    """
    return get_icon(f"menu/{category}/{name}.svg")


def get_icon_path(category: str, name: str) -> Optional[Path]:
    """
    Get the full path to an icon file.
    
    Args:
        category: Menu category
        name: Icon name without extension
        
    Returns:
        Path to icon file, or None if not found
    """
    icon_path = ICONS_BASE_PATH / "menu" / category / f"{name}.svg"
    return icon_path if icon_path.exists() else None


# Icon category constants for easy reference
class IconCategory:
    FILE = "file"
    SEARCH = "search"
    EDIT = "edit"
    VIEW = "view"
    TOOLS = "tools"
    PANELS = "panels"
    DATA_ANALYSIS = "data_analysis"
    MOUSE = "mouse"
    DRILLHOLES = "drillholes"
    RESOURCES = "resources"
    SCAN = "scan"
    ESTIMATIONS = "estimations"
    GEOLOGICAL_MODELING = "geological_modeling"
    GEOTECHNICAL = "geotechnical"
    MINE_PLANNING = "mine_planning"
    MACHINE_LEARNING = "machine_learning"
    DASHBOARDS = "dashboards"
    WORKBENCH = "workbench"
    WORKFLOWS = "workflows"
    HELP = "help"


# Convenience icon loaders for each category
class FileIcons:
    @staticmethod
    def new_project() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "new_project")
    
    @staticmethod
    def open_project() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "open_project")
    
    @staticmethod
    def save_project() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "save_project")
    
    @staticmethod
    def save_as() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "save_as")
    
    @staticmethod
    def open_file() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "open_file")
    
    @staticmethod
    def block_model_loading() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "block_model_loading")
    
    @staticmethod
    def recent_files() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "recent_files")
    
    @staticmethod
    def screenshot() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "screenshot")
    
    @staticmethod
    def export_data() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "export_data")
    
    @staticmethod
    def export_model() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "export_model")
    
    @staticmethod
    def restart() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "restart")
    
    @staticmethod
    def clear_scene() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "clear_scene")
    
    @staticmethod
    def exit() -> QIcon:
        return get_menu_icon(IconCategory.FILE, "exit")


class EditIcons:
    @staticmethod
    def undo() -> QIcon:
        return get_menu_icon(IconCategory.EDIT, "undo")
    
    @staticmethod
    def redo() -> QIcon:
        return get_menu_icon(IconCategory.EDIT, "redo")
    
    @staticmethod
    def preferences() -> QIcon:
        return get_menu_icon(IconCategory.EDIT, "preferences")


class ViewIcons:
    @staticmethod
    def reset_view() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "reset_view")
    
    @staticmethod
    def view_top() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_top")
    
    @staticmethod
    def view_bottom() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_bottom")
    
    @staticmethod
    def view_front() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_front")
    
    @staticmethod
    def view_back() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_back")
    
    @staticmethod
    def view_left() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_left")
    
    @staticmethod
    def view_right() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_right")
    
    @staticmethod
    def view_isometric() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "view_isometric")
    
    @staticmethod
    def orthographic() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "orthographic")
    
    @staticmethod
    def theme_light() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "theme_light")
    
    @staticmethod
    def theme_dark() -> QIcon:
        return get_menu_icon(IconCategory.VIEW, "theme_dark")


class ToolIcons:
    @staticmethod
    def slice() -> QIcon:
        return get_menu_icon(IconCategory.TOOLS, "slice")
    
    @staticmethod
    def filter() -> QIcon:
        return get_menu_icon(IconCategory.TOOLS, "filter")
    
    @staticmethod
    def selection() -> QIcon:
        return get_menu_icon(IconCategory.TOOLS, "selection")
    
    @staticmethod
    def statistics() -> QIcon:
        return get_menu_icon(IconCategory.TOOLS, "statistics")


class MouseIcons:
    @staticmethod
    def select() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "select")
    
    @staticmethod
    def pan() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "pan")
    
    @staticmethod
    def zoom() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "zoom")
    
    @staticmethod
    def zoom_in() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "zoom_in")
    
    @staticmethod
    def zoom_out() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "zoom_out")
    
    @staticmethod
    def zoom_extents() -> QIcon:
        return get_menu_icon(IconCategory.MOUSE, "zoom_extents")


class DrillholeIcons:
    @staticmethod
    def drillhole() -> QIcon:
        return get_menu_icon(IconCategory.DRILLHOLES, "drillhole")
    
    @staticmethod
    def loading() -> QIcon:
        return get_menu_icon(IconCategory.DRILLHOLES, "loading")
    
    @staticmethod
    def validation() -> QIcon:
        return get_menu_icon(IconCategory.DRILLHOLES, "validation")
    
    @staticmethod
    def compositing() -> QIcon:
        return get_menu_icon(IconCategory.DRILLHOLES, "compositing")
    
    @staticmethod
    def declustering() -> QIcon:
        return get_menu_icon(IconCategory.DRILLHOLES, "declustering")


class EstimationIcons:
    @staticmethod
    def variogram() -> QIcon:
        return get_menu_icon(IconCategory.ESTIMATIONS, "variogram")
    
    @staticmethod
    def kriging() -> QIcon:
        return get_menu_icon(IconCategory.ESTIMATIONS, "kriging")
    
    @staticmethod
    def simulation() -> QIcon:
        return get_menu_icon(IconCategory.ESTIMATIONS, "simulation")
    
    @staticmethod
    def uncertainty() -> QIcon:
        return get_menu_icon(IconCategory.ESTIMATIONS, "uncertainty")


class HelpIcons:
    @staticmethod
    def help() -> QIcon:
        return get_menu_icon(IconCategory.HELP, "help")
    
    @staticmethod
    def documentation() -> QIcon:
        return get_menu_icon(IconCategory.HELP, "documentation")
    
    @staticmethod
    def keyboard() -> QIcon:
        return get_menu_icon(IconCategory.HELP, "keyboard")
    
    @staticmethod
    def about() -> QIcon:
        return get_menu_icon(IconCategory.HELP, "about")

