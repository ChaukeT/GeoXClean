"""
Menu construction modules for GeoX.

Each menu file exports a single function:
    build_<menu_name>_menu(main_window: MainWindow, menubar: QMenuBar)

This keeps MainWindow clean and maintainable.
"""

from .file_menu import build_file_menu
from .search_menu import build_search_menu
from .edit_menu import build_edit_menu
from .view_menu import build_view_menu
from .survey_menu import build_survey_menu
from .scan_menu import build_scan_menu
from .remote_sensing_menu import build_remote_sensing_menu
from .tools_menu import build_tools_menu
from .panels_menu import build_panels_menu
from .data_menu import build_data_menu
from .mouse_menu import build_mouse_menu
from .drillholes_menu import build_drillholes_menu
from .geology_menu import build_geology_menu
from .resources_menu import build_resources_menu
from .estimations_menu import build_estimations_menu
from .geotech_menu import build_geotech_menu
from .mine_planning_menu import build_mine_planning_menu
from .ml_menu import build_ml_menu
from .dashboards_menu import build_dashboards_menu
from .workbench_menu import build_workbench_menu
from .workflows_menu import build_workflows_menu
from .help_menu import build_help_menu
from .layout_menu import build_layout_menu

__all__ = [
    'build_file_menu',
    'build_search_menu',
    'build_edit_menu',
    'build_view_menu',
    'build_survey_menu',
    'build_scan_menu',
    'build_remote_sensing_menu',
    'build_tools_menu',
    'build_panels_menu',
    'build_data_menu',
    'build_mouse_menu',
    'build_drillholes_menu',
    'build_geology_menu',
    'build_resources_menu',
    'build_estimations_menu',
    'build_geotech_menu',
    'build_mine_planning_menu',
    'build_ml_menu',
    'build_dashboards_menu',
    'build_workbench_menu',
    'build_workflows_menu',
    'build_help_menu',
    'build_layout_menu',
]

