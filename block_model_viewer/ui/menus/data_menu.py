"""
Data menu construction for GeoX.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction, QKeySequence

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None


def _safe_connect(action, main_window, method_name):
    """Connect an action's triggered signal to a main-window method, disabling the action if missing."""
    handler = getattr(main_window, method_name, None)
    if handler is not None:
        action.triggered.connect(handler)
    else:
        action.setEnabled(False)


def build_data_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Data menu."""
    data_menu = menubar.addMenu("&Data")

    # ── Block Model Construction ─────────────────────────────────
    builder_action = QAction(get_menu_icon("data", "block_model"), "&Block Model Builder...", main_window)
    builder_action.setStatusTip("Construct block models from geological and assay data")
    _safe_connect(builder_action, main_window, 'open_block_model_builder')
    data_menu.addAction(builder_action)

    calc_action = QAction(get_menu_icon("data", "calculator"), "Block &Property Calculator...", main_window)
    calc_action.setStatusTip("Compute derived block properties via expressions")
    _safe_connect(calc_action, main_window, 'open_block_property_calculator_panel')
    data_menu.addAction(calc_action)

    data_menu.addSeparator()

    # Statistics Window
    statistics_action = QAction(get_menu_icon("data_analysis", "statistics"), "Statistics", main_window)
    statistics_action.setStatusTip("Open statistics and data summary window")
    statistics_action.triggered.connect(main_window.open_statistics_window)
    data_menu.addAction(statistics_action)
    
    # Charts & Visualization Window
    charts_action = QAction(get_menu_icon("data_analysis", "charts"), "Charts && Visualization", main_window)
    charts_action.setStatusTip("Open charts and data visualization window")
    charts_action.triggered.connect(main_window.open_charts_window)
    data_menu.addAction(charts_action)
    
    # 3D Swath Analysis
    swath_3d_action = QAction(get_menu_icon("data_analysis", "swath_3d"), "3D Swath Analysis", main_window)
    swath_3d_action.setStatusTip("Geostatistical estimation reliability assessment using 3D swath analysis")
    swath_3d_action.triggered.connect(main_window.open_swath_analysis_3d_panel)
    data_menu.addAction(swath_3d_action)
    
    data_menu.addSeparator()
    
    # Structural Analysis
    structural_action = QAction(get_menu_icon("data_analysis", "structural"), "Structural Analysis...", main_window)
    structural_action.setStatusTip("Stereonet, rose diagrams, and kinematic feasibility analysis")
    structural_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
    structural_action.triggered.connect(main_window.open_structural_panel)
    data_menu.addAction(structural_action)

    # Swath Plot Analysis Window
    swath_action = QAction("Swath Plot Analysis", main_window)
    swath_action.setStatusTip("Open swath plot analysis with 3D linking")
    swath_action.triggered.connect(main_window.open_swath_window)
    data_menu.addAction(swath_action)

    data_menu.addSeparator()

    # ── Domain Modelling ──────────────────────────────────────────
    domain_menu = data_menu.addMenu("Domain Modelling")

    irbf_action = QAction(
        get_menu_icon("estimations", "rbf"),
        "&Indicator RBF Domain...",
        main_window,
    )
    irbf_action.setStatusTip(
        "Build implicit domain boundaries using Indicator Radial Basis Functions"
    )
    irbf_action.triggered.connect(main_window.open_indicator_rbf_panel)
    domain_menu.addAction(irbf_action)

    return data_menu