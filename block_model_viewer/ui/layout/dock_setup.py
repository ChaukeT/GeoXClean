"""Dock widget and toolbar setup for MainWindow."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from PyQt6.QtWidgets import QDockWidget, QTabWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon

if TYPE_CHECKING:
    from ..main_window import MainWindow

logger = logging.getLogger(__name__)


def setup_docks(main_window: 'MainWindow') -> None:
    """Setup all dockable panels.
    
    Args:
        main_window: MainWindow instance to configure docks for.
    """
    # LEFT DOCK - Tabbed (Property Controls + Scene Inspector)
    main_window.left_dock = QDockWidget("Controls & Scene", main_window)
    main_window.left_dock.setObjectName("Dock_ControlsScene")
    main_window.left_dock.setAllowedAreas(
        Qt.DockWidgetArea.LeftDockWidgetArea | 
        Qt.DockWidgetArea.RightDockWidgetArea
    )
    
    # Create tab widget for left dock
    main_window.left_tab_widget = QTabWidget()
    from ..scene_inspector_panel import SceneInspectorPanel
    main_window.scene_inspector_panel = SceneInspectorPanel(signals=main_window.signals)

    if main_window.controller:
        try:
            main_window.scene_inspector_panel.bind_controller(main_window.controller)
        except Exception:
            logger.debug("Failed to bind controller to display panels", exc_info=True)

    # Add tabs (Property Controls first - most important) with icons
    try:
        icons_dir = Path(__file__).parent.parent / 'icons'
        block_icon = QIcon(str(icons_dir / 'block.svg')) if (icons_dir / 'block.svg').exists() else QIcon()
        drill_icon = QIcon(str(icons_dir / 'drillhole.svg')) if (icons_dir / 'drillhole.svg').exists() else QIcon()
        pit_icon = QIcon(str(icons_dir / 'pit.svg')) if (icons_dir / 'pit.svg').exists() else QIcon()
    except Exception:
        block_icon = drill_icon = pit_icon = None

    if block_icon:
        main_window.left_tab_widget.addTab(main_window.property_panel, block_icon, "Property Controls")
    else:
        main_window.left_tab_widget.addTab(main_window.property_panel, "Property Controls")
    # Use pit icon for Scene Inspector (scene/ground/pit metaphor)
    if pit_icon:
        main_window.left_tab_widget.addTab(main_window.scene_inspector_panel, pit_icon, "Scene Inspector")
    else:
        main_window.left_tab_widget.addTab(main_window.scene_inspector_panel, "Scene Inspector")
    
    main_window.left_dock.setWidget(main_window.left_tab_widget)
    main_window.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, main_window.left_dock)
    
    # RIGHT DOCK - GC Decision Panel
    from ..gc_decision_panel import GCDecisionPanel
    main_window.gc_decision_panel = GCDecisionPanel()
    main_window.gc_decision_dock = QDockWidget("GC Decision Engine", main_window)
    main_window.gc_decision_dock.setObjectName("Dock_GCDecision")
    main_window.gc_decision_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
    main_window.gc_decision_dock.setWidget(main_window.gc_decision_panel)
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, main_window.gc_decision_dock)
    
    # RIGHT DOCK - Drillhole Controls
    from ..drillhole_control_panel import DrillholeControlPanel
    main_window.drillhole_control_panel = DrillholeControlPanel(signals=main_window.signals)
    # Signal connections are done in _connect_signals() method

    main_window.drillhole_control_dock = QDockWidget("Drillhole Explorer", main_window)
    main_window.drillhole_control_dock.setObjectName("Dock_DrillholeControls")
    main_window.drillhole_control_dock.setAllowedAreas(
        Qt.DockWidgetArea.LeftDockWidgetArea
        | Qt.DockWidgetArea.RightDockWidgetArea
        | Qt.DockWidgetArea.TopDockWidgetArea
        | Qt.DockWidgetArea.BottomDockWidgetArea
    )
    main_window.drillhole_control_dock.setWidget(main_window.drillhole_control_panel)
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, main_window.drillhole_control_dock)

    # RIGHT DOCK - Geological Explorer
    from ..geological_explorer_panel import GeologicalExplorerPanel
    main_window.geological_explorer_panel = GeologicalExplorerPanel(signals=main_window.signals)

    main_window.geological_explorer_dock = QDockWidget("Geological Explorer", main_window)
    main_window.geological_explorer_dock.setObjectName("Dock_GeologicalExplorer")
    main_window.geological_explorer_dock.setAllowedAreas(
        Qt.DockWidgetArea.LeftDockWidgetArea
        | Qt.DockWidgetArea.RightDockWidgetArea
        | Qt.DockWidgetArea.TopDockWidgetArea
        | Qt.DockWidgetArea.BottomDockWidgetArea
    )
    main_window.geological_explorer_dock.setWidget(main_window.geological_explorer_panel)
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, main_window.geological_explorer_dock)
    
    # Ensure both drillhole and geological explorer panels are tabified and visible
    main_window.tabifyDockWidget(main_window.drillhole_control_dock, main_window.geological_explorer_dock)
    
    # Show both panels at startup for better user experience
    main_window.drillhole_control_dock.show()
    main_window.geological_explorer_dock.show()
    
    # Raise drillhole explorer tab by default, but geological explorer will be readily accessible as a tab
    main_window.drillhole_control_dock.raise_()

    # Make all dock widgets hide-on-close (never destroy)
    main_window._setup_dock_hide_on_close(main_window.left_dock)
    main_window._setup_dock_hide_on_close(main_window.gc_decision_dock)
    main_window._setup_dock_hide_on_close(main_window.drillhole_control_dock)
    main_window._setup_dock_hide_on_close(main_window.geological_explorer_dock)
    
    # Connect registry signals to populate hole list (use injected registry)
    registry = main_window.controller.registry if main_window.controller else None
    if registry:
        try:
            if hasattr(registry, 'drillholeDataLoaded') and registry.drillholeDataLoaded:
                registry.drillholeDataLoaded.connect(main_window._on_drillhole_data_registered)
            # Connect DataRegistry.blockModelLoaded signal - single source of truth for viewer updates
            if hasattr(registry, 'blockModelLoaded') and registry.blockModelLoaded:
                registry.blockModelLoaded.connect(main_window._on_block_model_loaded_from_registry)
        except Exception as e:
            logger.error(f"Failed to connect registry signals in dock_setup: {e}", exc_info=True)
    
    # Connect scan registry signals for rendering
    if main_window.controller and hasattr(main_window.controller, 'scan_controller'):
        try:
            scan_registry = main_window.controller.scan_controller.registry
            if scan_registry and hasattr(scan_registry, 'signals'):
                scan_registry.signals.scanLoaded.connect(main_window._on_scan_loaded)
                logger.info("Connected to scan registry scanLoaded signal")
        except Exception as e:
            logger.debug(f"Could not connect to scan registry signals: {e}")
    
    # Keep references for compatibility (property_dock is now part of left_dock)
    main_window.scene_dock = main_window.left_dock  # For Panels menu
    main_window.property_dock = main_window.left_dock  # For compatibility
    
    # Setup camera update timer
    main_window.camera_update_timer = QTimer(main_window)
    main_window.camera_update_timer.timeout.connect(main_window._update_camera_info)
    main_window.camera_update_timer.start(500)  # Update every 500ms
    
    logger.info("Setup dockable panels")


def setup_toolbar(main_window: 'MainWindow') -> None:
    """Setup modern toolbar with icons, dropdowns, and status strip.
    
    Args:
        main_window: MainWindow instance to configure toolbar for.
    """
    from PyQt6.QtWidgets import QToolBar
    
    # Create modern toolbar widget
    from ..toolbar import Toolbar
    main_window.toolbar_widget = Toolbar()
    
    # Add toolbar widget to a QToolBar container
    toolbar = QToolBar("Main Toolbar", main_window)
    toolbar.setObjectName("main_toolbar")
    toolbar.setMovable(True)
    toolbar.setFloatable(False)
    toolbar.addWidget(main_window.toolbar_widget)
    main_window.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
    
    # Connect toolbar signals
    main_window.toolbar_widget.open_file_requested.connect(main_window.open_file)
    main_window.toolbar_widget.reset_view_requested.connect(main_window.reset_camera)
    main_window.toolbar_widget.fit_view_requested.connect(main_window.fit_to_view)
    main_window.toolbar_widget.export_screenshot_requested.connect(main_window.export_screenshot)
    main_window.toolbar_widget.export_data_requested.connect(main_window.export_filtered_data)
    
    # Connect new toolbar signals
    main_window.toolbar_widget.scene_action_requested.connect(main_window._handle_scene_action)
    main_window.toolbar_widget.view_action_requested.connect(main_window._handle_view_action)
    main_window.toolbar_widget.panel_action_requested.connect(main_window._handle_panel_action)
    
    logger.info("Setup modern toolbar")

